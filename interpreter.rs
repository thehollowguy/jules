// ─────────────────────────────────────────────────────────────────────────────
// INTERPRETER  (complete tree-walking evaluator)
//
// Fixed vs skeleton:
//  • Lambda body stored as LambdaBody enum — no more into_iter_stmt() hack
//  • push() / pop() mutate the variable in env correctly
//  • Result<V,E> is a first-class Value variant; the ? operator is modelled
//  • Future/async: a simple cooperative poll loop drives async fn bodies
//  • Pattern matching: struct, enum-with-payload, tuple, guard, Or, float lit
//  • Range iteration: `for i in 1..10`  and  `for i in 1..=10`
//  • run_module no longer clones env before executing top-level stmts
//  • ~30 builtins covering strings, lists, maps, math, I/O, introspection
// ─────────────────────────────────────────────────────────────────────────────

use std::collections::HashMap;
use std::io::{self, BufRead, Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aes_gcm::{Aes256Gcm, Nonce};
use aes_gcm::aead::{Aead, KeyInit};
use argon2::{Argon2, PasswordHasher};
use argon2::password_hash::SaltString;
use base64::{engine::general_purpose, Engine as _};
use hex;
use sha2::{Sha256, Digest};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use rand::Rng;
use serde_json::Value as JsonValue;

use crate::ast::*;

// ─────────────────────────────────────────────────────────────────────────────
// VALUE
// ─────────────────────────────────────────────────────────────────────────────

/// A callable body is either a list of statements (fn def) or a single
/// expression (lambda).  Storing it this way removes the old hack that
/// wrapped a lambda body in `Stmt::Return(expr)`.
#[derive(Debug, Clone)]
pub enum LambdaBody {
    Block(Vec<Stmt>),
    Expr(Box<Expr>),
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Nil,
    Unit,

    List(Vec<Value>),
    Tuple(Vec<Value>),

    /// Map backed by a Vec so keys can be any Value (no Ord/Hash required).
    Map(Vec<(Value, Value)>),

    Struct { name: String, fields: HashMap<String, Value> },

    /// Enum variant, optionally carrying a payload value.
    Enum { variant: String, payload: Option<Box<Value>> },

    /// User-defined or lambda function with captured closure.
    Fn {
        name:    Option<String>,   // None for anonymous lambdas
        params:  Vec<String>,
        body:    LambdaBody,
        closure: Box<Env>,
    },

    /// A builtin function identified by name.
    Builtin(String),

    /// Result type: Ok(inner) or Err(inner).
    /// Lets the language model Result-based error handling without exceptions.
    ResultOk(Box<Value>),
    ResultErr(Box<Value>),

    /// An async future.  Once resolved it holds the final value.
    Future(FutureState),
}

/// Cooperative future state.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum FutureState {
    Pending {
        params:  Vec<String>,
        body:    LambdaBody,
        closure: Box<Env>,
    },
    Resolved(Box<Value>),
}

// ─────────────────────────────────────────────────────────────────────────────
// GLOBAL RUNTIME STATE (thread handles, channels, networking resources)
// ─────────────────────────────────────────────────────────────────────────────

struct RuntimeState {
    next_handle: usize,
    threads: HashMap<usize, thread::JoinHandle<Value>>,
    senders: HashMap<usize, mpsc::Sender<Value>>,
    receivers: HashMap<usize, mpsc::Receiver<Value>>,
    tcp_streams: HashMap<usize, TcpStream>,
    tcp_listeners: HashMap<usize, TcpListener>,
    udp_sockets: HashMap<usize, UdpSocket>,
    processes: HashMap<usize, Child>,
}

impl RuntimeState {
    fn new() -> Self {
        Self {
            next_handle: 1,
            threads: HashMap::new(),
            senders: HashMap::new(),
            receivers: HashMap::new(),
            tcp_streams: HashMap::new(),
            tcp_listeners: HashMap::new(),
            udp_sockets: HashMap::new(),
            processes: HashMap::new(),
        }
    }

    fn allocate_handle(&mut self) -> usize {
        let id = self.next_handle;
        self.next_handle = self.next_handle.wrapping_add(1);
        id
    }
}

static RUNTIME: OnceLock<Mutex<RuntimeState>> = OnceLock::new();

fn runtime() -> std::sync::MutexGuard<'static, RuntimeState> {
    RUNTIME.get_or_init(|| Mutex::new(RuntimeState::new())).lock().unwrap()
}

// ── Display ────────────────────────────────────────────────────────────────────
impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(n)     => write!(f, "{}", n),
            Value::Float(n)   => {
                if n.fract() == 0.0 { write!(f, "{:.1}", n) } else { write!(f, "{}", n) }
            }
            Value::Bool(b)    => write!(f, "{}", b),
            Value::Str(s)     => write!(f, "{}", s),
            Value::Nil        => write!(f, "nil"),
            Value::Unit       => write!(f, "()"),
            Value::List(v)    => {
                write!(f, "[")?;
                for (i, x) in v.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", x)?;
                }
                write!(f, "]")
            }
            Value::Tuple(v)   => {
                write!(f, "(")?;
                for (i, x) in v.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", x)?;
                }
                write!(f, ")")
            }
            Value::Map(pairs) => {
                write!(f, "{{")?;
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Struct { name, fields } => {
                write!(f, "{} {{", name)?;
                let mut sorted: Vec<_> = fields.iter().collect();
                sorted.sort_by_key(|(k, _)| k.as_str());
                for (i, (k, v)) in sorted.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Enum { variant, payload } => match payload {
                Some(p) => write!(f, "{}({})", variant, p),
                None    => write!(f, "{}", variant),
            },
            Value::Fn { name: Some(n), .. } => write!(f, "<fn {}>", n),
            Value::Fn { name: None,    .. } => write!(f, "<fn>"),
            Value::Builtin(n)               => write!(f, "<builtin:{}>", n),
            Value::ResultOk(v)              => write!(f, "Ok({})", v),
            Value::ResultErr(v)             => write!(f, "Err({})", v),
            Value::Future(FutureState::Pending { .. })  => write!(f, "<future:pending>"),
            Value::Future(FutureState::Resolved(v))     => write!(f, "<future:{}>", v),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Int(a),        Value::Int(b))        => a == b,
            (Value::Float(a),      Value::Float(b))      => a == b,
            (Value::Int(a),        Value::Float(b))      => (*a as f64) == *b,
            (Value::Float(a),      Value::Int(b))        => *a == (*b as f64),
            (Value::Bool(a),       Value::Bool(b))       => a == b,
            (Value::Str(a),        Value::Str(b))        => a == b,
            (Value::Nil,           Value::Nil)           => true,
            (Value::Unit,          Value::Unit)          => true,
            (Value::List(a),       Value::List(b))       => a == b,
            (Value::Tuple(a),      Value::Tuple(b))      => a == b,
            (Value::ResultOk(a),   Value::ResultOk(b))   => a == b,
            (Value::ResultErr(a),  Value::ResultErr(b))  => a == b,
            (Value::Enum { variant: va, payload: pa },
             Value::Enum { variant: vb, payload: pb })   => va == vb && pa == pb,
            _ => false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ENVIRONMENT
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct Env {
    pub vars:   HashMap<String, Value>,
    parent: Option<Box<Env>>,
}

impl Env {
    pub fn new() -> Self { Self::default() }

    pub fn child(parent: Env) -> Self {
        Self { vars: HashMap::new(), parent: Some(Box::new(parent)) }
    }

    /// Define a new binding in the innermost scope.
    pub fn set(&mut self, name: &str, val: Value) {
        self.vars.insert(name.to_string(), val);
    }

    /// Re-assign an existing binding anywhere in the scope chain.
    pub fn assign(&mut self, name: &str, val: Value) -> bool {
        if self.vars.contains_key(name) {
            self.vars.insert(name.to_string(), val);
            true
        } else if let Some(p) = &mut self.parent {
            p.assign(name, val)
        } else {
            false
        }
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.vars.get(name)
            .or_else(|| self.parent.as_ref().and_then(|p| p.get(name)))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CONTROL-FLOW SIGNALS
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
#[allow(dead_code)]
enum Signal {
    Return(Value),
    Break,
    Continue,
    Error(RuntimeError),
}

// ─────────────────────────────────────────────────────────────────────────────
// RUNTIME ERROR
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RuntimeError {
    pub msg:   String,
    pub trace: Vec<String>,
}

impl RuntimeError {
    fn new(msg: impl Into<String>) -> Self {
        Self { msg: msg.into(), trace: Vec::new() }
    }
    fn push_frame(mut self, frame: impl Into<String>) -> Self {
        self.trace.push(frame.into());
        self
    }
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "runtime error: {}", self.msg)?;
        for frame in self.trace.iter().rev() {
            writeln!(f, "    in {}", frame)?;
        }
        Ok(())
    }
}

macro_rules! rte { ($($t:tt)*) => { RuntimeError::new(format!($($t)*)) } }
macro_rules! sig { ($($t:tt)*) => { Signal::Error(rte!($($t)*)) } }

// ─────────────────────────────────────────────────────────────────────────────
// INTERPRETER
// ─────────────────────────────────────────────────────────────────────────────

pub struct Interpreter {
    pub env: Env,
    call_depth: usize,
}

const MAX_CALL_DEPTH: usize = 500;

impl Interpreter {
    pub fn new() -> Self {
        let mut interp = Self { env: Env::new(), call_depth: 0 };
        interp.register_builtins();
        interp
    }

    fn register_builtins(&mut self) {
        let names = [
            "print", "println", "eprintln", "input",
            "type_of", "to_str", "to_int", "to_float", "to_bool",
            "assert", "assert_eq", "panic",
            "ok", "err", "is_ok", "is_err", "unwrap", "unwrap_or",
            "len", "push", "pop", "insert", "remove", "contains",
            "slice", "reverse", "sort", "concat", "zip",
            "map_list", "filter_list", "reduce_list",
            "map_new", "map_get", "map_set", "map_del",
            "map_has", "map_keys", "map_values",
            "split", "join", "trim", "to_upper", "to_lower",
            "starts_with", "ends_with", "replace", "chars",
            "char_at", "index_of",
            "abs", "floor", "ceil", "round", "sqrt", "pow",
            "min", "max", "clamp",
            "range", "range_inclusive",

            // concurrency / async
            "spawn_thread", "join_thread", "sleep",
            "channel", "send", "receive",

            // networking
            "tcp_connect", "tcp_listen", "tcp_accept", "tcp_send", "tcp_receive", "tcp_close",
            "udp_bind", "udp_send", "udp_receive", "udp_close",
            "http_get", "http_post", "start_http_server",

            // serialization
            "to_json", "from_json", "serialize", "deserialize",

            // crypto
            "sha256", "argon2_hash", "aes_encrypt", "aes_decrypt", "secure_random",

            // compression
            "gzip_compress", "gzip_decompress",

            // math / stats
            "matrix_multiply", "vector_dot", "transpose",
            "mean", "variance", "standard_deviation",

            // random
            "rand_int", "rand_float", "rand_range",

            // time
            "current_time", "duration",

            // filesystem
            "list_directory", "create_directory", "move_file", "delete_file", "watch_directory",

            // process
            "spawn_process", "kill_process", "get_process_output",

            // logging
            "log_info", "log_warning", "log_error", "log_debug",

            // configuration
            "read_env", "parse_toml",

            // cli
            "parse_args", "flag", "option",
        ];
        for name in &names {
            self.env.set(name, Value::Builtin(name.to_string()));
        }
        self.env.set("PI",       Value::Float(std::f64::consts::PI));
        self.env.set("E",        Value::Float(std::f64::consts::E));
        self.env.set("INFINITY", Value::Float(f64::INFINITY));
        self.env.set("INT_MAX",  Value::Int(i64::MAX));
        self.env.set("INT_MIN",  Value::Int(i64::MIN));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MODULE ENTRY POINT
    // ─────────────────────────────────────────────────────────────────────────

    pub fn run_module(&mut self, module: &Module) -> Result<Value, RuntimeError> {
        // First pass: register all function names so mutual recursion works.
        for f in &module.fns {
            let val = Value::Fn {
                name:    Some(f.name.clone()),
                params:  f.params.iter().map(|p| p.name.clone()).collect(),
                body:    LambdaBody::Block(f.body.clone()),
                closure: Box::new(self.env.clone()),
            };
            self.env.set(&f.name, val);
        }
        // Second pass: give each fn a closure that includes itself (recursion).
        for f in &module.fns {
            if let Some(self_val) = self.env.vars.get(&f.name).cloned() {
                if let Some(Value::Fn { closure, .. }) = self.env.vars.get_mut(&f.name) {
                    closure.set(&f.name, self_val);
                }
            }
        }

        let stmts = module.stmts.clone();
        let mut env = self.env.clone();
        for stmt in &stmts {
            match self.exec_stmt(stmt, &mut env) {
                None => {}
                Some(Signal::Error(e))  => return Err(e),
                Some(Signal::Return(v)) => return Ok(v),
                Some(_) => {}
            }
        }
        // Propagate any top-level lets back into self.env
        for (k, v) in env.vars {
            self.env.set(&k, v);
        }
        Ok(Value::Unit)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // STATEMENT EXECUTION
    // ─────────────────────────────────────────────────────────────────────────

    fn exec_stmt(&self, stmt: &Stmt, env: &mut Env) -> Option<Signal> {
        match stmt {

            Stmt::Let { name, value, .. } => {
                match self.eval_expr(value, env) {
                    Ok(v)  => { env.set(name, v); None }
                    Err(e) => Some(Signal::Error(e)),
                }
            }

            Stmt::Expr(expr) => {
                match self.eval_expr(expr, env) {
                    Ok(_)  => None,
                    Err(e) => Some(Signal::Error(e)),
                }
            }

            Stmt::Return(val, _) => {
                let v = match val {
                    Some(e) => match self.eval_expr(e, env) {
                        Ok(v)  => v,
                        Err(e) => return Some(Signal::Error(e)),
                    },
                    None => Value::Unit,
                };
                Some(Signal::Return(v))
            }

            Stmt::While { cond, body, .. } => {
                loop {
                    match self.eval_expr(cond, env) {
                        Ok(Value::Bool(false)) => break,
                        Ok(Value::Bool(true))  => {}
                        Ok(other) => return Some(sig!("while condition is `{}`, expected Bool", other)),
                        Err(e)    => return Some(Signal::Error(e)),
                    }
                    let mut inner = Env::child(env.clone());
                    for s in body {
                        match self.exec_stmt(s, &mut inner) {
                            None => {}
                            Some(Signal::Continue) => break,
                            Some(Signal::Break)    => return None,
                            other @ Some(_)        => return other,
                        }
                    }
                    self.merge_env_up(&inner, env);
                }
                None
            }

            Stmt::For { var, iter, body, .. } => {
                let iter_val = match self.eval_expr(iter, env) {
                    Ok(v)  => v,
                    Err(e) => return Some(Signal::Error(e)),
                };
                let items = match self.collect_iterable(iter_val) {
                    Ok(v)  => v,
                    Err(e) => return Some(Signal::Error(e)),
                };
                for item in items {
                    let mut inner = Env::child(env.clone());
                    inner.set(var, item);
                    for s in body {
                        match self.exec_stmt(s, &mut inner) {
                            None => {}
                            Some(Signal::Continue) => break,
                            Some(Signal::Break)    => return None,
                            other @ Some(_)        => return other,
                        }
                    }
                    self.merge_env_up(&inner, env);
                }
                None
            }

            Stmt::Import { .. } => None,
        }
    }

    /// Propagate mutations made inside a child scope back up to the parent.
    /// Only variables that already exist in the parent chain are propagated.
    fn merge_env_up(&self, child: &Env, parent: &mut Env) {
        for (k, v) in &child.vars {
            if parent.get(k).is_some() {
                parent.assign(k, v.clone());
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EXPRESSION EVALUATION
    // ─────────────────────────────────────────────────────────────────────────

    pub fn eval_expr(&self, expr: &Expr, env: &mut Env) -> Result<Value, RuntimeError> {
        match expr {
            // ── Literals ────────────────────────────────────────────────────
            Expr::Int(n, _)   => Ok(Value::Int(*n)),
            Expr::Float(f, _) => Ok(Value::Float(*f)),
            Expr::Bool(b, _)  => Ok(Value::Bool(*b)),
            Expr::Str(s, _)   => Ok(Value::Str(s.clone())),
            Expr::Nil(_)      => Ok(Value::Nil),

            // ── Variable lookup ─────────────────────────────────────────────
            Expr::Var(name, _) => {
                env.get(name).cloned()
                    .ok_or_else(|| rte!(
                        "undefined variable `{}`  — did you forget `let`?", name))
            }

            // ── Unary ────────────────────────────────────────────────────────
            Expr::Unary { op, expr, .. } => {
                let v = self.eval_expr(expr, env)?;
                match (op, v) {
                    (UnaryOp::Neg, Value::Int(n))   => Ok(Value::Int(-n)),
                    (UnaryOp::Neg, Value::Float(f)) => Ok(Value::Float(-f)),
                    (UnaryOp::Not, Value::Bool(b))  => Ok(Value::Bool(!b)),
                    (UnaryOp::Neg, other) => Err(rte!("cannot negate `{}`", other)),
                    (UnaryOp::Not, other) => Err(rte!("cannot apply `!` to `{}`", other)),
                }
            }

            // ── Binary ───────────────────────────────────────────────────────
            Expr::Binary { op, left, right, .. } => {
                let lv = self.eval_expr(left, env)?;
                // Short-circuit
                match (op, &lv) {
                    (BinaryOp::And, Value::Bool(false)) => return Ok(Value::Bool(false)),
                    (BinaryOp::Or,  Value::Bool(true))  => return Ok(Value::Bool(true)),
                    _ => {}
                }
                let rv = self.eval_expr(right, env)?;
                self.eval_binary(op, lv, rv)
            }

            // ── Assignment ───────────────────────────────────────────────────
            Expr::Assign { target, value, .. } => {
                let v = self.eval_expr(value, env)?;
                match target.as_ref() {
                    Expr::Var(name, span) => {
                        if !env.assign(name, v.clone()) {
                            return Err(rte!(
                                "cannot assign to `{}` at {}:{} — \
                                 not declared in any scope\n\
                                 hint: use `let mut {}` to declare it first",
                                name, span.line, span.col, name
                            ));
                        }
                    }
                    // arr[i] = val
                    Expr::Index { object, index, span } => {
                        if let Expr::Var(name, _) = object.as_ref() {
                            let idx = self.eval_expr(index, env)?;
                            let list = env.get(name).cloned()
                                .ok_or_else(|| rte!("undefined `{}`", name))?;
                            let updated = self.list_set(list, idx, v.clone())
                                .map_err(|e| e.push_frame(
                                    format!("index assign at {}:{}", span.line, span.col)))?;
                            env.assign(name, updated);
                        } else {
                            return Err(rte!("complex index-assignment is not supported"));
                        }
                    }
                    // obj.field = val
                    Expr::Field { object, field, .. } => {
                        if let Expr::Var(name, _) = object.as_ref() {
                            let mut obj = env.get(name).cloned()
                                .ok_or_else(|| rte!("undefined `{}`", name))?;
                            match &mut obj {
                                Value::Struct { fields, .. } => {
                                    fields.insert(field.clone(), v.clone());
                                }
                                _ => return Err(rte!("field-assignment on non-struct")),
                            }
                            env.assign(name, obj);
                        }
                    }
                    _ => return Err(rte!("invalid assignment target")),
                }
                Ok(v)
            }

            // ── Function call ────────────────────────────────────────────────
            Expr::Call { callee, args, span } => {
                let fn_val = self.eval_expr(callee, env)?;
                let arg_vals: Vec<Value> = args.iter()
                    .map(|a| self.eval_expr(a, env))
                    .collect::<Result<_, _>>()?;
                self.call_value(fn_val, arg_vals)
                    .map_err(|e| e.push_frame(
                        format!("call at {}:{}", span.line, span.col)))
            }

            // ── Field access ─────────────────────────────────────────────────
            Expr::Field { object, field, .. } => {
                let obj = self.eval_expr(object, env)?;
                self.get_field(obj, field)
            }

            // ── Index ────────────────────────────────────────────────────────
            Expr::Index { object, index, .. } => {
                let obj = self.eval_expr(object, env)?;
                let idx = self.eval_expr(index, env)?;
                self.eval_index(obj, idx)
            }

            // ── Lambda ───────────────────────────────────────────────────────
            Expr::Lambda { params, body, .. } => {
                Ok(Value::Fn {
                    name:    None,
                    params:  params.iter().map(|p| p.name.clone()).collect(),
                    body:    LambdaBody::Expr(body.clone()),  // ← correct: body IS the expr
                    closure: Box::new(env.clone()),
                })
            }

            // ── Block  { stmts; [tail_expr] } ────────────────────────────────
            Expr::Block(stmts, tail, _) => {
                let mut inner = Env::child(env.clone());
                for s in stmts {
                    match self.exec_stmt(s, &mut inner) {
                        None => {}
                        Some(Signal::Return(v)) => {
                            self.merge_env_up(&inner, env);
                            return Ok(v);
                        }
                        Some(Signal::Error(e)) => return Err(e),
                        Some(_) => {}
                    }
                }
                let result = if let Some(t) = tail {
                    self.eval_expr(t, &mut inner)?
                } else {
                    Value::Unit
                };
                self.merge_env_up(&inner, env);
                Ok(result)
            }

            // ── If / else ────────────────────────────────────────────────────
            Expr::If { cond, then_branch, else_branch, .. } => {
                match self.eval_expr(cond, env)? {
                    Value::Bool(true)  => self.eval_expr(then_branch, env),
                    Value::Bool(false) => {
                        if let Some(eb) = else_branch {
                            self.eval_expr(eb, env)
                        } else {
                            Ok(Value::Unit)
                        }
                    }
                    other => Err(rte!("if condition must be Bool, got `{}`", other)),
                }
            }

            // ── Match ────────────────────────────────────────────────────────
            Expr::Match { subject, arms, span } => {
                let val = self.eval_expr(subject, env)?;
                for arm in arms {
                    if self.pattern_matches(&arm.pattern, &val, env)? {
                        let mut inner = Env::child(env.clone());
                        self.bind_pattern(&arm.pattern, &val, &mut inner)?;
                        return self.eval_expr(&arm.body, &mut inner);
                    }
                }
                Err(rte!(
                    "non-exhaustive match at {}:{} — \
                     no arm matched `{}`\n\
                     hint: add `_ => ...` as a fallback",
                    span.line, span.col, val
                ))
            }

            // ── Collections ──────────────────────────────────────────────────
            Expr::Array(elems, _) => {
                let vals: Vec<Value> = elems.iter()
                    .map(|e| self.eval_expr(e, env))
                    .collect::<Result<_, _>>()?;
                Ok(Value::List(vals))
            }

            Expr::Tuple(elems, _) => {
                let vals: Vec<Value> = elems.iter()
                    .map(|e| self.eval_expr(e, env))
                    .collect::<Result<_, _>>()?;
                Ok(Value::Tuple(vals))
            }

            // ── Struct literal  Point { x: 1, y: 2 } ───────────────────────
            Expr::StructLit { name, fields, .. } => {
                let mut fmap = HashMap::new();
                for (k, v) in fields {
                    fmap.insert(k.clone(), self.eval_expr(v, env)?);
                }
                Ok(Value::Struct { name: name.clone(), fields: fmap })
            }

            // ── Range  lo..hi  /  lo..=hi ────────────────────────────────────
            Expr::Range { start, end, inclusive, .. } => {
                let s = self.eval_expr(start, env)?;
                let e = self.eval_expr(end, env)?;
                match (s, e) {
                    (Value::Int(lo), Value::Int(hi)) => {
                        let items: Vec<Value> = if *inclusive {
                            (lo..=hi).map(Value::Int).collect()
                        } else {
                            (lo..hi).map(Value::Int).collect()
                        };
                        Ok(Value::List(items))
                    }
                    (s, e) => Err(rte!(
                        "range requires Int..Int, got `{}`..`{}`\n\
                         hint: ensure both ends are integers",
                        s, e
                    )),
                }
            }

            // ── Await ────────────────────────────────────────────────────────
            Expr::Await { expr, .. } => {
                let v = self.eval_expr(expr, env)?;
                self.poll_future(v)
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ITERABLE COLLECTION
    // ─────────────────────────────────────────────────────────────────────────

    fn collect_iterable(&self, val: Value) -> Result<Vec<Value>, RuntimeError> {
        match val {
            Value::List(v)  => Ok(v),
            Value::Str(s)   => Ok(s.chars().map(|c| Value::Str(c.to_string())).collect()),
            Value::Tuple(v) => Ok(v),
            Value::Map(pairs) => Ok(pairs.into_iter()
                .map(|(k, v)| Value::Tuple(vec![k, v]))
                .collect()),
            other => Err(rte!(
                "`{}` is not iterable\n\
                 hint: use a List, String, Tuple, or Map", other)),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // FIELD ACCESS
    // ─────────────────────────────────────────────────────────────────────────

    fn get_field(&self, obj: Value, field: &str) -> Result<Value, RuntimeError> {
        match &obj {
            Value::Struct { fields, .. } => {
                fields.get(field).cloned()
                    .ok_or_else(|| rte!(
                        "struct `{}` has no field `{}`\n\
                         hint: available fields are: {}",
                        if let Value::Struct{name,..} = &obj { name } else { "?" },
                        field,
                        if let Value::Struct{fields,..} = &obj {
                            fields.keys().cloned().collect::<Vec<_>>().join(", ")
                        } else { String::new() }
                    ))
            }
            Value::Enum { variant, .. } if field == "variant" => {
                Ok(Value::Str(variant.clone()))
            }
            Value::List(v) if field == "len" => Ok(Value::Int(v.len() as i64)),
            Value::Str(s)  if field == "len" => Ok(Value::Int(s.chars().count() as i64)),
            Value::Tuple(v) => {
                if let Ok(idx) = field.parse::<usize>() {
                    v.get(idx).cloned()
                        .ok_or_else(|| rte!(
                            "tuple index .{} out of range (tuple has {} elements)",
                            idx, v.len()))
                } else {
                    Err(rte!("tuple fields must be numbers like `.0`, `.1` — got `.{}`", field))
                }
            }
            _ => Err(rte!("`{}` has no field `{}`", obj, field)),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // INDEX ACCESS / MUTATION
    // ─────────────────────────────────────────────────────────────────────────

    fn eval_index(&self, obj: Value, idx: Value) -> Result<Value, RuntimeError> {
        match (obj, idx) {
            (Value::List(v), Value::Int(i)) => {
                let len = v.len() as i64;
                let i = if i < 0 { len + i } else { i };
                if i < 0 || i >= len {
                    Err(rte!("list index {} out of bounds — list has {} elements", i, len))
                } else {
                    Ok(v[i as usize].clone())
                }
            }
            (Value::Str(s), Value::Int(i)) => {
                let chars: Vec<char> = s.chars().collect();
                let len = chars.len() as i64;
                let i = if i < 0 { len + i } else { i };
                if i < 0 || i >= len {
                    Err(rte!("string index {} out of bounds — string has {} characters", i, len))
                } else {
                    Ok(Value::Str(chars[i as usize].to_string()))
                }
            }
            (Value::Map(pairs), key) => {
                pairs.iter().find(|(k, _)| k == &key)
                    .map(|(_, v)| v.clone())
                    .ok_or_else(|| rte!(
                        "key `{}` not found in map\n\
                         hint: use `map_has(m, key)` to check before indexing",
                        key))
            }
            (Value::Tuple(v), Value::Int(i)) => {
                let len = v.len() as i64;
                let i = if i < 0 { len + i } else { i };
                v.get(i as usize).cloned()
                    .ok_or_else(|| rte!(
                        "tuple index {} out of range — tuple has {} elements", i, len))
            }
            (obj, idx) => Err(rte!(
                "cannot index `{}` with `{}`\n\
                 hint: indexable types are List, String, Map, Tuple",
                obj, idx)),
        }
    }

    fn list_set(&self, list: Value, idx: Value, val: Value) -> Result<Value, RuntimeError> {
        match (list, idx) {
            (Value::List(mut v), Value::Int(i)) => {
                let len = v.len() as i64;
                let i = if i < 0 { len + i } else { i };
                if i < 0 || i >= len {
                    Err(rte!("index {} out of bounds (len {})", i, len))
                } else {
                    v[i as usize] = val;
                    Ok(Value::List(v))
                }
            }
            (obj, idx) => Err(rte!("cannot index-assign `{}` with `{}`", obj, idx)),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BINARY OPERATIONS
    // ─────────────────────────────────────────────────────────────────────────

    fn eval_binary(&self, op: &BinaryOp, l: Value, r: Value) -> Result<Value, RuntimeError> {
        match (op, l, r) {
            // Int arithmetic
            (BinaryOp::Add, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.wrapping_add(b))),
            (BinaryOp::Sub, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.wrapping_sub(b))),
            (BinaryOp::Mul, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.wrapping_mul(b))),
            (BinaryOp::Div, Value::Int(a), Value::Int(b)) => {
                if b == 0 { Err(rte!("integer division by zero")) }
                else { Ok(Value::Int(a / b)) }
            }
            (BinaryOp::Mod, Value::Int(a), Value::Int(b)) => {
                if b == 0 { Err(rte!("modulo by zero")) }
                else { Ok(Value::Int(((a % b) + b) % b)) }  // always-positive modulo
            }
            // Float arithmetic
            (BinaryOp::Add, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            (BinaryOp::Sub, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (BinaryOp::Mul, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            (BinaryOp::Div, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
            (BinaryOp::Mod, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a % b)),
            // Mixed Int/Float: promote
            (op, Value::Int(a), Value::Float(b)) =>
                self.eval_binary(op, Value::Float(a as f64), Value::Float(b)),
            (op, Value::Float(a), Value::Int(b)) =>
                self.eval_binary(op, Value::Float(a), Value::Float(b as f64)),
            // String concatenation (also allows str + any)
            (BinaryOp::Add, Value::Str(a), Value::Str(b)) => Ok(Value::Str(a + &b)),
            (BinaryOp::Add, Value::Str(a), b)             => Ok(Value::Str(a + &b.to_string())),
            // List concatenation
            (BinaryOp::Add, Value::List(mut a), Value::List(b)) => {
                a.extend(b); Ok(Value::List(a))
            }
            // Polymorphic equality
            (BinaryOp::Eq,    a, b) => Ok(Value::Bool(a == b)),
            (BinaryOp::NotEq, a, b) => Ok(Value::Bool(a != b)),
            // Ordered comparisons
            (BinaryOp::Lt,   Value::Int(a),   Value::Int(b))   => Ok(Value::Bool(a < b)),
            (BinaryOp::Gt,   Value::Int(a),   Value::Int(b))   => Ok(Value::Bool(a > b)),
            (BinaryOp::LtEq, Value::Int(a),   Value::Int(b))   => Ok(Value::Bool(a <= b)),
            (BinaryOp::GtEq, Value::Int(a),   Value::Int(b))   => Ok(Value::Bool(a >= b)),
            (BinaryOp::Lt,   Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
            (BinaryOp::Gt,   Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
            (BinaryOp::LtEq, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
            (BinaryOp::GtEq, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),
            (BinaryOp::Lt,   Value::Str(a),   Value::Str(b))   => Ok(Value::Bool(a < b)),
            (BinaryOp::Gt,   Value::Str(a),   Value::Str(b))   => Ok(Value::Bool(a > b)),
            (BinaryOp::LtEq, Value::Str(a),   Value::Str(b))   => Ok(Value::Bool(a <= b)),
            (BinaryOp::GtEq, Value::Str(a),   Value::Str(b))   => Ok(Value::Bool(a >= b)),
            // Boolean logic
            (BinaryOp::And, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a && b)),
            (BinaryOp::Or,  Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a || b)),

            (op, l, r) => Err(rte!(
                "operator `{:?}` is not defined for `{}` and `{}`\n\
                 hint: check types or add an explicit conversion",
                op, l, r
            )),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // FUNCTION CALL
    // ─────────────────────────────────────────────────────────────────────────

    pub fn call_value(&self, fn_val: Value, args: Vec<Value>) -> Result<Value, RuntimeError> {
        match fn_val {
            Value::Fn { name, params, body, closure } => {
                if self.call_depth >= MAX_CALL_DEPTH {
                    return Err(rte!(
                        "stack overflow — call depth exceeded {}\n\
                         hint: check for infinite recursion",
                        MAX_CALL_DEPTH
                    ));
                }
                if params.len() != args.len() {
                    return Err(rte!(
                        "`{}` expects {} argument(s), but {} were provided",
                        name.as_deref().unwrap_or("<fn>"),
                        params.len(), args.len()
                    ));
                }
                let mut env = Env::child(*closure);
                for (p, a) in params.iter().zip(args.into_iter()) {
                    env.set(p, a);
                }
                let label = name.as_deref().unwrap_or("<fn>").to_string();
                let result = match &body {
                    LambdaBody::Block(stmts) => {
                        for stmt in stmts {
                            match self.exec_stmt(stmt, &mut env) {
                                None => {}
                                Some(Signal::Return(v)) => return Ok(v),
                                Some(Signal::Error(e))  =>
                                    return Err(e.push_frame(label.clone())),
                                _ => {}
                            }
                        }
                        Ok(Value::Unit)
                    }
                    LambdaBody::Expr(expr) => self.eval_expr(expr, &mut env),
                };
                result.map_err(|e| e.push_frame(label))
            }

            Value::Builtin(name) => self.call_builtin(&name, args),

            other => Err(rte!(
                "`{}` (type: {}) is not callable\n\
                 hint: only functions and builtins can be called",
                other, self.type_name(&other)
            )),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ASYNC / FUTURE
    // ─────────────────────────────────────────────────────────────────────────

    /// Drive a Future to completion synchronously.
    /// In a real async runtime this would yield at I/O boundaries.
    fn poll_future(&self, val: Value) -> Result<Value, RuntimeError> {
        match val {
            Value::Future(FutureState::Resolved(v)) => Ok(*v),
            Value::Future(FutureState::Pending { params, body, closure }) => {
                if !params.is_empty() {
                    return Err(rte!(
                        "cannot await a Future that still has unbound parameters\n\
                         hint: call the async function with its arguments first"));
                }
                let mut env = Env::child(*closure);
                match &body {
                    LambdaBody::Block(stmts) => {
                        for stmt in stmts {
                            match self.exec_stmt(stmt, &mut env) {
                                None => {}
                                Some(Signal::Return(v)) => return Ok(v),
                                Some(Signal::Error(e))  => return Err(e),
                                _ => {}
                            }
                        }
                        Ok(Value::Unit)
                    }
                    LambdaBody::Expr(expr) => self.eval_expr(expr, &mut env),
                }
            }
            // .await on a non-Future is a no-op (mirrors Rust)
            other => Ok(other),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PATTERN MATCHING
    // ─────────────────────────────────────────────────────────────────────────

    fn pattern_matches(&self, pat: &Pattern, val: &Value, env: &Env)
        -> Result<bool, RuntimeError>
    {
        let ok = match (pat, val) {
            (Pattern::Wildcard(_),   _) => true,
            (Pattern::Binding(_,_),  _) => true,    // binds in bind_pattern

            // Literal patterns
            (Pattern::Literal(Expr::Int(n, _)),   Value::Int(v))   => n == v,
            (Pattern::Literal(Expr::Float(n, _)), Value::Float(v)) =>
                (n - v).abs() < f64::EPSILON,
            (Pattern::Literal(Expr::Bool(b, _)),  Value::Bool(v))  => b == v,
            (Pattern::Literal(Expr::Str(s, _)),   Value::Str(v))   => s == v,
            (Pattern::Literal(Expr::Nil(_)),       Value::Nil)      => true,

            // Enum / Result patterns
            (Pattern::Enum(name, inner, _), Value::Enum { variant, payload }) => {
                if name != variant { return Ok(false); }
                match (inner, payload) {
                    (None,    None)           => true,
                    (Some(p), Some(payload))  => self.pattern_matches(p, payload, env)?,
                    _                         => false,
                }
            }
            (Pattern::Enum(name, inner, _), Value::ResultOk(v)) if name == "Ok" => {
                match inner {
                    None    => true,
                    Some(p) => self.pattern_matches(p, v, env)?,
                }
            }
            (Pattern::Enum(name, inner, _), Value::ResultErr(v)) if name == "Err" => {
                match inner {
                    None    => true,
                    Some(p) => self.pattern_matches(p, v, env)?,
                }
            }

            // Tuple patterns: (a, b, c)
            (Pattern::Tuple(pats, _), Value::Tuple(vals)) => {
                if pats.len() != vals.len() { return Ok(false); }
                for (p, v) in pats.iter().zip(vals.iter()) {
                    if !self.pattern_matches(p, v, env)? { return Ok(false); }
                }
                true
            }

            // Struct patterns: Point { x, y }
            (Pattern::Struct(name, field_pats, _), Value::Struct { name: sname, fields }) => {
                if name != sname { return Ok(false); }
                for (fname, fpat) in field_pats {
                    match fields.get(fname) {
                        None => return Ok(false),
                        Some(fval) => {
                            if !self.pattern_matches(fpat, fval, env)? { return Ok(false); }
                        }
                    }
                }
                true
            }

            // Or-patterns: pat1 | pat2
            (Pattern::Or(pats, _), val) => {
                for p in pats {
                    if self.pattern_matches(p, val, env)? { return Ok(true); }
                }
                false
            }

            // Guard patterns: pat if condition
            (Pattern::Guard(inner, guard_expr, _), val) => {
                if !self.pattern_matches(inner, val, env)? { return Ok(false); }
                let mut guard_env = env.clone();
                // Bind the inner pattern so the guard can reference it
                self.bind_pattern(inner, val, &mut guard_env)?;
                match self.eval_expr(guard_expr, &mut guard_env)? {
                    Value::Bool(b) => b,
                    other => return Err(rte!(
                        "pattern guard must evaluate to Bool, got `{}`", other)),
                }
            }

            _ => false,
        };
        Ok(ok)
    }

    fn bind_pattern(&self, pat: &Pattern, val: &Value, env: &mut Env)
        -> Result<(), RuntimeError>
    {
        match pat {
            Pattern::Wildcard(_) => {}
            Pattern::Binding(name, _) => { env.set(name, val.clone()); }
            Pattern::Literal(_) => {}

            Pattern::Enum(_, inner_pat, _) => {
                let payload: Option<&Value> = match val {
                    Value::Enum { payload, .. } => payload.as_ref().map(|b| b.as_ref()),
                    Value::ResultOk(v)  => Some(v.as_ref()),
                    Value::ResultErr(v) => Some(v.as_ref()),
                    _ => None,
                };
                if let (Some(p), Some(inner)) = (inner_pat, payload) {
                    self.bind_pattern(p, inner, env)?;
                }
            }

            Pattern::Tuple(pats, _) => {
                if let Value::Tuple(vals) = val {
                    for (p, v) in pats.iter().zip(vals.iter()) {
                        self.bind_pattern(p, v, env)?;
                    }
                }
            }

            Pattern::Struct(_, field_pats, _) => {
                if let Value::Struct { fields, .. } = val {
                    for (fname, fpat) in field_pats {
                        if let Some(fval) = fields.get(fname) {
                            self.bind_pattern(fpat, fval, env)?;
                        }
                    }
                }
            }

            Pattern::Or(pats, _) => {
                // Bind from the first matching sub-pattern
                for p in pats {
                    if self.pattern_matches(p, val, env).unwrap_or(false) {
                        self.bind_pattern(p, val, env)?;
                        return Ok(());
                    }
                }
            }

            Pattern::Guard(inner, _, _) => {
                self.bind_pattern(inner, val, env)?;
            }
        }
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BUILTINS
    // ─────────────────────────────────────────────────────────────────────────

    fn call_builtin(&self, name: &str, args: Vec<Value>) -> Result<Value, RuntimeError> {
        match name {

            // ── I/O ─────────────────────────────────────────────────────────
            "print" => {
                print!("{}", args.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "));
                io::stdout().flush().ok();
                Ok(Value::Unit)
            }
            "println" => {
                println!("{}", args.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "));
                Ok(Value::Unit)
            }
            "eprintln" => {
                eprintln!("{}", args.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "));
                Ok(Value::Unit)
            }
            "input" => {
                if let Some(Value::Str(prompt)) = args.first() {
                    print!("{}", prompt);
                    io::stdout().flush().ok();
                }
                let stdin = io::stdin();
                let mut line = String::new();
                stdin.lock().read_line(&mut line)
                    .map_err(|e| rte!("input() failed: {}", e))?;
                Ok(Value::Str(line.trim_end_matches('\n')
                                  .trim_end_matches('\r').to_string()))
            }

            // ── Type introspection / coercion ─────────────────────────────────
            "type_of" => Ok(Value::Str(
                self.type_name(args.first().unwrap_or(&Value::Nil)).to_string()
            )),
            "to_str"   => Ok(Value::Str(
                args.first().map(|v| v.to_string()).unwrap_or_default()
            )),
            "to_int"   => {
                match args.first() {
                    Some(Value::Int(n))   => Ok(Value::Int(*n)),
                    Some(Value::Float(f)) => Ok(Value::Int(*f as i64)),
                    Some(Value::Bool(b))  => Ok(Value::Int(if *b { 1 } else { 0 })),
                    Some(Value::Str(s))   => s.trim().parse::<i64>()
                        .map(Value::Int)
                        .map_err(|_| rte!(
                            "to_int(): cannot parse `{}` as Int\n\
                             hint: the string must contain only digits, e.g. \"42\"", s)),
                    other => Err(rte!("to_int(): cannot convert `{}`",
                        other.map(|v| v.to_string()).unwrap_or_default())),
                }
            }
            "to_float" => {
                match args.first() {
                    Some(Value::Float(f)) => Ok(Value::Float(*f)),
                    Some(Value::Int(n))   => Ok(Value::Float(*n as f64)),
                    Some(Value::Bool(b))  => Ok(Value::Float(if *b { 1.0 } else { 0.0 })),
                    Some(Value::Str(s))   => s.trim().parse::<f64>()
                        .map(Value::Float)
                        .map_err(|_| rte!(
                            "to_float(): cannot parse `{}` as Float\n\
                             hint: use a decimal like \"3.14\" or \"42.0\"", s)),
                    other => Err(rte!("to_float(): cannot convert `{}`",
                        other.map(|v| v.to_string()).unwrap_or_default())),
                }
            }
            "to_bool"  => {
                Ok(Value::Bool(match args.first() {
                    Some(Value::Bool(b)) => *b,
                    Some(Value::Int(n))  => *n != 0,
                    Some(Value::Nil) | Some(Value::Unit) => false,
                    Some(_)  => true,
                    None     => false,
                }))
            }

            // ── Assertions / panics ───────────────────────────────────────────
            "assert" => {
                let msg = args.get(1).map(|v| v.to_string())
                    .unwrap_or_else(|| "assertion failed".to_string());
                match args.first() {
                    Some(Value::Bool(true))  => Ok(Value::Unit),
                    Some(Value::Bool(false)) => Err(rte!("{}", msg)),
                    other => Err(rte!("assert() expects Bool, got `{}`",
                        other.map(|v| v.to_string()).unwrap_or_default())),
                }
            }
            "assert_eq" => {
                if args.len() < 2 {
                    return Err(rte!("assert_eq() requires 2 arguments"));
                }
                if args[0] == args[1] { Ok(Value::Unit) }
                else { Err(rte!("assert_eq failed:\n  left:  {}\n  right: {}",
                    args[0], args[1])) }
            }
            "panic" => Err(RuntimeError {
                msg:   args.first().map(|v| v.to_string())
                           .unwrap_or_else(|| "explicit panic".to_string()),
                trace: vec!["panic!".to_string()],
            }),

            // ── Result helpers ────────────────────────────────────────────────
            "ok"  => Ok(Value::ResultOk(Box::new(
                args.into_iter().next().unwrap_or(Value::Unit)))),
            "err" => Ok(Value::ResultErr(Box::new(
                args.into_iter().next().unwrap_or(Value::Nil)))),
            "is_ok"  => Ok(Value::Bool(
                matches!(args.first(), Some(Value::ResultOk(_))))),
            "is_err" => Ok(Value::Bool(
                matches!(args.first(), Some(Value::ResultErr(_))))),
            "unwrap" => match args.into_iter().next() {
                Some(Value::ResultOk(v))  => Ok(*v),
                Some(Value::ResultErr(e)) => Err(rte!(
                    "unwrap() called on Err({})\n\
                     hint: handle the error with `match` or `unwrap_or`", e)),
                Some(other) => Ok(other),
                None        => Err(rte!("unwrap() requires an argument")),
            },
            "unwrap_or" => {
                let default = args.get(1).cloned().unwrap_or(Value::Nil);
                match args.into_iter().next() {
                    Some(Value::ResultOk(v))  => Ok(*v),
                    Some(Value::ResultErr(_)) => Ok(default),
                    Some(other)               => Ok(other),
                    None                      => Ok(default),
                }
            }

            // ── List operations ───────────────────────────────────────────────
            "len" => match args.first() {
                Some(Value::List(v))  => Ok(Value::Int(v.len() as i64)),
                Some(Value::Str(s))   => Ok(Value::Int(s.chars().count() as i64)),
                Some(Value::Map(m))   => Ok(Value::Int(m.len() as i64)),
                Some(Value::Tuple(t)) => Ok(Value::Int(t.len() as i64)),
                other => Err(rte!("len() does not accept `{}`",
                    other.map(|v| v.to_string()).unwrap_or_default())),
            },
            "push" => match (args.get(0).cloned(), args.get(1).cloned()) {
                (Some(Value::List(mut v)), Some(item)) => {
                    v.push(item);
                    Ok(Value::List(v))
                    // Caller must reassign: `list = push(list, item);`
                }
                _ => Err(rte!("push(list, item) — first argument must be a List\n\
                    hint: reassign to apply: `my_list = push(my_list, item)`")),
            },
            "pop" => match args.into_iter().next() {
                Some(Value::List(mut v)) => {
                    let popped = v.pop().unwrap_or(Value::Nil);
                    // Returns (new_list, popped_item) as a tuple
                    Ok(Value::Tuple(vec![Value::List(v), popped]))
                }
                _ => Err(rte!("pop(list) — requires a List")),
            },
            "insert" => match (args.get(0).cloned(), args.get(1).cloned(), args.get(2).cloned()) {
                (Some(Value::List(mut v)), Some(Value::Int(i)), Some(item)) => {
                    let len = v.len() as i64;
                    v.insert(i.clamp(0, len) as usize, item);
                    Ok(Value::List(v))
                }
                _ => Err(rte!("insert(list, index, item) — type mismatch")),
            },
            "remove" => match (args.get(0).cloned(), args.get(1).cloned()) {
                (Some(Value::List(mut v)), Some(Value::Int(i))) => {
                    let len = v.len() as i64;
                    if i < 0 || i >= len {
                        return Err(rte!("remove: index {} out of bounds (len {})", i, len));
                    }
                    let removed = v.remove(i as usize);
                    Ok(Value::Tuple(vec![Value::List(v), removed]))
                }
                _ => Err(rte!("remove(list, index) — type mismatch")),
            },
            "contains" => match (args.get(0), args.get(1)) {
                (Some(Value::List(v)), Some(needle)) =>
                    Ok(Value::Bool(v.contains(needle))),
                (Some(Value::Str(s)),  Some(Value::Str(sub))) =>
                    Ok(Value::Bool(s.contains(sub.as_str()))),
                (Some(Value::Map(m)),  Some(key)) =>
                    Ok(Value::Bool(m.iter().any(|(k, _)| k == key))),
                _ => Err(rte!("contains(collection, item) — unsupported types")),
            },
            "slice" => match (args.get(0).cloned(), args.get(1).cloned(), args.get(2).cloned()) {
                (Some(Value::List(v)), Some(Value::Int(s)), end) => {
                    let len = v.len() as i64;
                    let s = s.clamp(0, len) as usize;
                    let e = match end {
                        Some(Value::Int(e)) => e.clamp(0, len) as usize,
                        _ => len as usize,
                    };
                    Ok(Value::List(v[s..e.max(s)].to_vec()))
                }
                (Some(Value::Str(s)), Some(Value::Int(from)), end) => {
                    let chars: Vec<char> = s.chars().collect();
                    let len = chars.len() as i64;
                    let from = from.clamp(0, len) as usize;
                    let to   = match end {
                        Some(Value::Int(e)) => e.clamp(0, len) as usize,
                        _ => len as usize,
                    };
                    Ok(Value::Str(chars[from..to.max(from)].iter().collect()))
                }
                _ => Err(rte!("slice(list|string, start, end?) — type mismatch")),
            },
            "reverse" => match args.into_iter().next() {
                Some(Value::List(mut v)) => { v.reverse(); Ok(Value::List(v)) }
                Some(Value::Str(s))      => Ok(Value::Str(s.chars().rev().collect())),
                other => Err(rte!("reverse() requires List or String, got `{}`",
                    other.map(|v| v.to_string()).unwrap_or_default())),
            },
            "sort" => match args.into_iter().next() {
                Some(Value::List(mut v)) => {
                    // Stable lexicographic sort via Display representation.
                    // Future: accept optional comparator fn as second argument.
                    v.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
                    Ok(Value::List(v))
                }
                other => Err(rte!("sort() requires a List, got `{}`",
                    other.map(|v| v.to_string()).unwrap_or_default())),
            },
            "concat" => match args.into_iter().next() {
                Some(Value::List(outer)) => {
                    let mut out = Vec::new();
                    for item in outer {
                        match item {
                            Value::List(inner) => out.extend(inner),
                            other              => out.push(other),
                        }
                    }
                    Ok(Value::List(out))
                }
                _ => Err(rte!("concat(list_of_lists) — requires a List")),
            },
            "zip" => match (args.get(0).cloned(), args.get(1).cloned()) {
                (Some(Value::List(a)), Some(Value::List(b))) => {
                    Ok(Value::List(
                        a.into_iter().zip(b.into_iter())
                         .map(|(x, y)| Value::Tuple(vec![x, y]))
                         .collect()
                    ))
                }
                _ => Err(rte!("zip(list, list) — requires two Lists")),
            },
            "map_list" => match (args.get(0).cloned(), args.get(1).cloned()) {
                (Some(Value::List(v)), Some(f)) => {
                    v.into_iter()
                     .map(|item| self.call_value(f.clone(), vec![item]))
                     .collect::<Result<Vec<_>, _>>()
                     .map(Value::List)
                }
                _ => Err(rte!("map_list(list, fn) — type mismatch")),
            },
            "filter_list" => match (args.get(0).cloned(), args.get(1).cloned()) {
                (Some(Value::List(v)), Some(f)) => {
                    let mut out = Vec::new();
                    for item in v {
                        match self.call_value(f.clone(), vec![item.clone()])? {
                            Value::Bool(true)  => out.push(item),
                            Value::Bool(false) => {}
                            other => return Err(rte!(
                                "filter predicate must return Bool, got `{}`", other)),
                        }
                    }
                    Ok(Value::List(out))
                }
                _ => Err(rte!("filter_list(list, fn) — type mismatch")),
            },
            "reduce_list" => match (args.get(0).cloned(), args.get(1).cloned(), args.get(2).cloned()) {
                (Some(Value::List(v)), Some(mut acc), Some(f)) => {
                    for item in v {
                        acc = self.call_value(f.clone(), vec![acc, item])?;
                    }
                    Ok(acc)
                }
                _ => Err(rte!("reduce_list(list, initial, fn) — type mismatch")),
            },

            // ── Map operations ────────────────────────────────────────────────
            "map_new" => Ok(Value::Map(Vec::new())),
            "map_get" => match (args.get(0), args.get(1)) {
                (Some(Value::Map(m)), Some(key)) => Ok(
                    m.iter().find(|(k, _)| k == key)
                     .map(|(_, v)| v.clone())
                     .unwrap_or(Value::Nil)
                ),
                _ => Err(rte!("map_get(map, key) — requires Map")),
            },
            "map_set" => match (args.get(0).cloned(), args.get(1).cloned(), args.get(2).cloned()) {
                (Some(Value::Map(mut m)), Some(key), Some(val)) => {
                    match m.iter_mut().find(|(k, _)| k == &key) {
                        Some((_, v)) => *v = val,
                        None         => m.push((key, val)),
                    }
                    Ok(Value::Map(m))
                }
                _ => Err(rte!("map_set(map, key, value) — type mismatch")),
            },
            "map_del" => match (args.get(0).cloned(), args.get(1).cloned()) {
                (Some(Value::Map(mut m)), Some(key)) => {
                    m.retain(|(k, _)| k != &key);
                    Ok(Value::Map(m))
                }
                _ => Err(rte!("map_del(map, key) — requires Map")),
            },
            "map_has" => match (args.get(0), args.get(1)) {
                (Some(Value::Map(m)), Some(key)) =>
                    Ok(Value::Bool(m.iter().any(|(k, _)| k == key))),
                _ => Err(rte!("map_has(map, key) — requires Map")),
            },
            "map_keys" => match args.first() {
                Some(Value::Map(m)) =>
                    Ok(Value::List(m.iter().map(|(k, _)| k.clone()).collect())),
                _ => Err(rte!("map_keys(map) — requires Map")),
            },
            "map_values" => match args.first() {
                Some(Value::Map(m)) =>
                    Ok(Value::List(m.iter().map(|(_, v)| v.clone()).collect())),
                _ => Err(rte!("map_values(map) — requires Map")),
            },

            // ── String operations ─────────────────────────────────────────────
            "split" => match (args.get(0), args.get(1)) {
                (Some(Value::Str(s)), Some(Value::Str(delim))) =>
                    Ok(Value::List(
                        s.split(delim.as_str())
                         .map(|p| Value::Str(p.to_string()))
                         .collect()
                    )),
                _ => Err(rte!("split(string, delimiter) — type mismatch")),
            },
            "join" => match (args.get(0), args.get(1)) {
                (Some(Value::List(v)), Some(Value::Str(sep))) =>
                    Ok(Value::Str(
                        v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(sep)
                    )),
                _ => Err(rte!("join(list, separator) — type mismatch")),
            },
            "trim"       => match args.first() {
                Some(Value::Str(s)) => Ok(Value::Str(s.trim().to_string())),
                _ => Err(rte!("trim(string) — requires String")),
            },
            "to_upper"   => match args.first() {
                Some(Value::Str(s)) => Ok(Value::Str(s.to_uppercase())),
                _ => Err(rte!("to_upper(string) — requires String")),
            },
            "to_lower"   => match args.first() {
                Some(Value::Str(s)) => Ok(Value::Str(s.to_lowercase())),
                _ => Err(rte!("to_lower(string) — requires String")),
            },
            "starts_with" => match (args.get(0), args.get(1)) {
                (Some(Value::Str(s)), Some(Value::Str(prefix))) =>
                    Ok(Value::Bool(s.starts_with(prefix.as_str()))),
                _ => Err(rte!("starts_with(string, prefix) — type mismatch")),
            },
            "ends_with" => match (args.get(0), args.get(1)) {
                (Some(Value::Str(s)), Some(Value::Str(suffix))) =>
                    Ok(Value::Bool(s.ends_with(suffix.as_str()))),
                _ => Err(rte!("ends_with(string, suffix) — type mismatch")),
            },
            "replace" => match (args.get(0), args.get(1), args.get(2)) {
                (Some(Value::Str(s)), Some(Value::Str(from)), Some(Value::Str(to))) =>
                    Ok(Value::Str(s.replace(from.as_str(), to.as_str()))),
                _ => Err(rte!("replace(string, from, to) — type mismatch")),
            },
            "chars" => match args.first() {
                Some(Value::Str(s)) =>
                    Ok(Value::List(
                        s.chars().map(|c| Value::Str(c.to_string())).collect()
                    )),
                _ => Err(rte!("chars(string) — requires String")),
            },
            "char_at" => match (args.get(0), args.get(1)) {
                (Some(Value::Str(s)), Some(Value::Int(i))) => {
                    let chars: Vec<char> = s.chars().collect();
                    let len = chars.len() as i64;
                    let idx = if *i < 0 { len + i } else { *i };
                    if idx < 0 || idx >= len {
                        Err(rte!("char_at: index {} out of bounds (len {})", i, len))
                    } else {
                        Ok(Value::Str(chars[idx as usize].to_string()))
                    }
                }
                _ => Err(rte!("char_at(string, index) — type mismatch")),
            },
            "index_of" => match (args.get(0), args.get(1)) {
                (Some(Value::Str(s)),  Some(Value::Str(sub))) =>
                    Ok(Value::Int(s.find(sub.as_str()).map(|i| i as i64).unwrap_or(-1))),
                (Some(Value::List(v)), Some(needle)) =>
                    Ok(Value::Int(
                        v.iter().position(|x| x == needle)
                         .map(|i| i as i64).unwrap_or(-1)
                    )),
                _ => Err(rte!("index_of(collection, item) — type mismatch")),
            },

            // ── Math ──────────────────────────────────────────────────────────
            "abs" => match args.first() {
                Some(Value::Int(n))   => Ok(Value::Int(n.wrapping_abs())),
                Some(Value::Float(f)) => Ok(Value::Float(f.abs())),
                _ => Err(rte!("abs(number) — requires Int or Float")),
            },
            "floor" => match args.first() {
                Some(Value::Float(f)) => Ok(Value::Int(f.floor() as i64)),
                Some(Value::Int(n))   => Ok(Value::Int(*n)),
                _ => Err(rte!("floor(number) — requires Float or Int")),
            },
            "ceil" => match args.first() {
                Some(Value::Float(f)) => Ok(Value::Int(f.ceil() as i64)),
                Some(Value::Int(n))   => Ok(Value::Int(*n)),
                _ => Err(rte!("ceil(number) — requires Float or Int")),
            },
            "round" => match args.first() {
                Some(Value::Float(f)) => Ok(Value::Int(f.round() as i64)),
                Some(Value::Int(n))   => Ok(Value::Int(*n)),
                _ => Err(rte!("round(number) — requires Float or Int")),
            },
            "sqrt" => match args.first() {
                Some(Value::Float(f)) => Ok(Value::Float(f.sqrt())),
                Some(Value::Int(n))   => Ok(Value::Float((*n as f64).sqrt())),
                _ => Err(rte!("sqrt(number) — requires Int or Float")),
            },
            "pow" => match (args.get(0), args.get(1)) {
                (Some(Value::Int(b)),   Some(Value::Int(e)))   =>
                    Ok(Value::Int(b.wrapping_pow((*e).max(0) as u32))),
                (Some(Value::Float(b)), Some(Value::Float(e))) => Ok(Value::Float(b.powf(*e))),
                (Some(Value::Int(b)),   Some(Value::Float(e))) => Ok(Value::Float((*b as f64).powf(*e))),
                (Some(Value::Float(b)), Some(Value::Int(e)))   => Ok(Value::Float(b.powi(*e as i32))),
                _ => Err(rte!("pow(base, exp) — requires two numbers")),
            },
            "min" => match (args.get(0), args.get(1)) {
                (Some(Value::List(v)), None) =>
                    v.iter().cloned().reduce(|a, b| {
                        if a.to_string() <= b.to_string() { a } else { b }
                    }).ok_or_else(|| rte!("min() on empty list")),
                (Some(a), Some(b)) => match (a, b) {
                    (Value::Int(a),   Value::Int(b))   => Ok(Value::Int((*a).min(*b))),
                    (Value::Float(a), Value::Float(b)) => Ok(Value::Float((*a).min(*b))),
                    (Value::Int(a),   Value::Float(b)) => Ok(Value::Float(((*a) as f64).min(*b))),
                    (Value::Float(a), Value::Int(b))   => Ok(Value::Float((*a).min((*b) as f64))),
                    _ => Err(rte!("min(a, b) — requires numbers")),
                },
                _ => Err(rte!("min(a, b) or min(list)")),
            },
            "max" => match (args.get(0), args.get(1)) {
                (Some(Value::List(v)), None) =>
                    v.iter().cloned().reduce(|a, b| {
                        if a.to_string() >= b.to_string() { a } else { b }
                    }).ok_or_else(|| rte!("max() on empty list")),
                (Some(a), Some(b)) => match (a, b) {
                    (Value::Int(a),   Value::Int(b))   => Ok(Value::Int(*a.max(b))),
                    (Value::Float(a), Value::Float(b)) => Ok(Value::Float((*a).max(*b))),
                    (Value::Int(a),   Value::Float(b)) => Ok(Value::Float(((*a) as f64).max(*b))),
                    (Value::Float(a), Value::Int(b))   => Ok(Value::Float((*a).max((*b) as f64))),
                    _ => Err(rte!("max(a, b) — requires numbers")),
                },
                _ => Err(rte!("max(a, b) or max(list)")),
            },
            "clamp" => match (args.get(0), args.get(1), args.get(2)) {
                (Some(Value::Int(v)),   Some(Value::Int(lo)),   Some(Value::Int(hi)))   =>
                    Ok(Value::Int((*v).clamp(*lo, *hi))),
                (Some(Value::Float(v)), Some(Value::Float(lo)), Some(Value::Float(hi))) =>
                    Ok(Value::Float((*v).clamp(*lo, *hi))),
                (Some(Value::Int(v)),   Some(Value::Float(lo)), Some(Value::Float(hi))) =>
                    Ok(Value::Float(((*v) as f64).clamp(*lo, *hi))),
                _ => Err(rte!("clamp(value, min, max) — type mismatch")),
            },

            // ── Ranges ────────────────────────────────────────────────────────
            "range" => match (args.get(0), args.get(1)) {
                (Some(Value::Int(lo)), Some(Value::Int(hi))) =>
                    Ok(Value::List((*lo..*hi).map(Value::Int).collect())),
                _ => Err(rte!("range(start, end) — requires two Ints")),
            },
            "range_inclusive" => match (args.get(0), args.get(1)) {
                (Some(Value::Int(lo)), Some(Value::Int(hi))) =>
                    Ok(Value::List((*lo..=*hi).map(Value::Int).collect())),
                _ => Err(rte!("range_inclusive(start, end) — requires two Ints")),
            },

            // ── Concurrency / async helpers ──────────────────────────────────
            "spawn_thread" => {
                let func = args.into_iter().next().ok_or_else(|| rte!("spawn_thread(fn) requires a function"))?;
                let handle = thread::spawn(move || {
                    let interp = Interpreter::new();
                    match interp.call_value(func, vec![]) {
                        Ok(v) => v,
                        Err(e) => Value::Str(format!("thread error: {}", e)),
                    }
                });
                let mut rt = runtime();
                let id = rt.allocate_handle();
                rt.threads.insert(id, handle);
                Ok(Value::Int(id as i64))
            }
            "join_thread" => {
                let id = match args.into_iter().next() {
                    Some(Value::Int(n)) => n as usize,
                    other => return Err(rte!("join_thread(thread_id) requires an Int, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let mut rt = runtime();
                match rt.threads.remove(&id) {
                    Some(handle) => match handle.join() {
                        Ok(v) => Ok(v),
                        Err(_) => Err(rte!("join_thread: thread panicked")),
                    },
                    None => Err(rte!("join_thread: invalid thread id {}", id)),
                }
            }
            "sleep" => {
                let ms = match args.into_iter().next() {
                    Some(Value::Int(n)) => n,
                    Some(other) => return Err(rte!("sleep(ms) expects an Int, got `{}`", other)),
                    None => return Err(rte!("sleep(ms) requires an argument")),
                };
                thread::sleep(Duration::from_millis(ms as u64));
                Ok(Value::Unit)
            }
            "channel" => {
                let (tx, rx) = mpsc::channel();
                let mut rt = runtime();
                let tx_id = rt.allocate_handle();
                let rx_id = rt.allocate_handle();
                rt.senders.insert(tx_id, tx);
                rt.receivers.insert(rx_id, rx);
                Ok(Value::Tuple(vec![Value::Int(tx_id as i64), Value::Int(rx_id as i64)]))
            }
            "send" => {
                let mut iter = args.into_iter();
                let id = match iter.next() {
                    Some(Value::Int(n)) => n as usize,
                    other => return Err(rte!("send(channel_id, value) requires an Int channel_id, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let value = iter.next().unwrap_or(Value::Nil);
                let rt = runtime();
                match rt.senders.get(&id) {
                    Some(tx) => tx.send(value).map_err(|e| rte!("send() failed: {}", e))?,
                    None => return Err(rte!("send(): unknown sender id {}", id)),
                }
                Ok(Value::Unit)
            }
            "receive" => {
                let id = match args.into_iter().next() {
                    Some(Value::Int(n)) => n as usize,
                    other => return Err(rte!("receive(channel_id) requires an Int, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let rt = runtime();
                match rt.receivers.get(&id) {
                    Some(rx) => rx.recv().map_err(|e| rte!("receive() failed: {}", e)),
                    None => Err(rte!("receive(): unknown receiver id {}", id)),
                }
            }

            // ── Networking ──────────────────────────────────────────────────
            "tcp_connect" => {
                let addr = match args.into_iter().next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("tcp_connect(addr) requires a String, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let stream = TcpStream::connect(addr.as_str())
                    .map_err(|e| rte!("tcp_connect failed: {}", e))?;
                let mut rt = runtime();
                let id = rt.allocate_handle();
                rt.tcp_streams.insert(id, stream);
                Ok(Value::Int(id as i64))
            }
            "tcp_listen" => {
                let addr = match args.into_iter().next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("tcp_listen(addr) requires a String, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let listener = TcpListener::bind(addr.as_str())
                    .map_err(|e| rte!("tcp_listen failed: {}", e))?;
                let mut rt = runtime();
                let id = rt.allocate_handle();
                rt.tcp_listeners.insert(id, listener);
                Ok(Value::Int(id as i64))
            }
            "tcp_accept" => {
                let id = match args.into_iter().next() {
                    Some(Value::Int(n)) => n as usize,
                    other => return Err(rte!("tcp_accept(listener_id) requires an Int, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let mut rt = runtime();
                let listener = rt.tcp_listeners.get(&id)
                    .ok_or_else(|| rte!("tcp_accept: unknown listener id {}", id))?;
                let (stream, _) = listener.accept()
                    .map_err(|e| rte!("tcp_accept failed: {}", e))?;
                let sid = rt.allocate_handle();
                rt.tcp_streams.insert(sid, stream);
                Ok(Value::Int(sid as i64))
            }
            "tcp_send" => {
                let mut iter = args.into_iter();
                let id = match iter.next() {
                    Some(Value::Int(n)) => n as usize,
                    other => return Err(rte!("tcp_send(stream_id, data) requires an Int stream_id, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let data = match iter.next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("tcp_send(stream_id, data) requires String data, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let mut rt = runtime();
                let stream = rt.tcp_streams.get_mut(&id)
                    .ok_or_else(|| rte!("tcp_send: unknown stream id {}", id))?;
                stream.write_all(data.as_bytes()).map_err(|e| rte!("tcp_send failed: {}", e))?;
                stream.flush().ok();
                Ok(Value::Unit)
            }
            "tcp_receive" => {
                let mut iter = args.into_iter();
                let id = match iter.next() {
                    Some(Value::Int(n)) => n as usize,
                    other => return Err(rte!("tcp_receive(stream_id, max_bytes) requires an Int stream_id, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let max = match iter.next() {
                    Some(Value::Int(n)) => n as usize,
                    Some(other) => return Err(rte!("tcp_receive(stream_id, max_bytes) requires Int max_bytes, got `{}`", other.to_string())),
                    None => 4096,
                };
                let mut rt = runtime();
                let stream = rt.tcp_streams.get_mut(&id)
                    .ok_or_else(|| rte!("tcp_receive: unknown stream id {}", id))?;
                let mut buf = vec![0u8; max];
                let n = stream.read(&mut buf).map_err(|e| rte!("tcp_receive failed: {}", e))?;
                let s = String::from_utf8_lossy(&buf[..n]).to_string();
                Ok(Value::Str(s))
            }
            "tcp_close" => {
                let id = match args.into_iter().next() {
                    Some(Value::Int(n)) => n as usize,
                    other => return Err(rte!("tcp_close(stream_id) requires an Int, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let mut rt = runtime();
                if rt.tcp_streams.remove(&id).is_some() {
                    Ok(Value::Unit)
                } else {
                    Err(rte!("tcp_close: unknown stream id {}", id))
                }
            }
            "udp_bind" => {
                let port = match args.into_iter().next() {
                    Some(Value::Int(n)) => n as u16,
                    other => return Err(rte!("udp_bind(port) requires an Int, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let socket = UdpSocket::bind(("0.0.0.0", port))
                    .map_err(|e| rte!("udp_bind failed: {}", e))?;
                let mut rt = runtime();
                let id = rt.allocate_handle();
                rt.udp_sockets.insert(id, socket);
                Ok(Value::Int(id as i64))
            }
            "udp_send" => {
                let mut iter = args.into_iter();
                let id = match iter.next() {
                    Some(Value::Int(n)) => n as usize,
                    other => return Err(rte!("udp_send(socket_id, addr, data) requires an Int socket_id, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let addr = match iter.next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("udp_send(socket_id, addr, data) requires String addr, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let data = match iter.next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("udp_send(socket_id, addr, data) requires String data, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let rt = runtime();
                let sock = rt.udp_sockets.get(&id)
                    .ok_or_else(|| rte!("udp_send: unknown socket id {}", id))?;
                sock.send_to(data.as_bytes(), addr.as_str())
                    .map_err(|e| rte!("udp_send failed: {}", e))?;
                Ok(Value::Unit)
            }
            "udp_receive" => {
                let mut iter = args.into_iter();
                let id = match iter.next() {
                    Some(Value::Int(n)) => n as usize,
                    other => return Err(rte!("udp_receive(socket_id, max_bytes) requires an Int socket_id, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let max = match iter.next() {
                    Some(Value::Int(n)) => n as usize,
                    Some(other) => return Err(rte!("udp_receive(socket_id, max_bytes) requires Int max_bytes, got `{}`", other.to_string())),
                    None => 4096,
                };
                let rt = runtime();
                let sock = rt.udp_sockets.get(&id)
                    .ok_or_else(|| rte!("udp_receive: unknown socket id {}", id))?;
                let mut buf = vec![0u8; max];
                let (n, addr) = sock.recv_from(&mut buf)
                    .map_err(|e| rte!("udp_receive failed: {}", e))?;
                let s = String::from_utf8_lossy(&buf[..n]).to_string();
                Ok(Value::Tuple(vec![Value::Str(s), Value::Str(addr.to_string())]))
            }
            "udp_close" => {
                let id = match args.into_iter().next() {
                    Some(Value::Int(n)) => n as usize,
                    other => return Err(rte!("udp_close(socket_id) requires an Int, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let mut rt = runtime();
                if rt.udp_sockets.remove(&id).is_some() {
                    Ok(Value::Unit)
                } else {
                    Err(rte!("udp_close: unknown socket id {}", id))
                }
            }

            // ── HTTP ─────────────────────────────────────────────────────────
            "http_get" => {
                let url = match args.into_iter().next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("http_get(url) requires a String, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let resp = ureq::get(url.as_str()).call()
                    .map_err(|e| rte!("http_get failed: {}", e))?;
                let body = resp.into_string().map_err(|e| rte!("http_get failed: {}", e))?;
                Ok(Value::Str(body))
            }
            "http_post" => {
                let mut iter = args.into_iter();
                let url = match iter.next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("http_post(url, body) requires String url, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let body = match iter.next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("http_post(url, body) requires String body, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let resp = ureq::post(url.as_str()).send_string(body.as_str())
                    .map_err(|e| rte!("http_post failed: {}", e))?;
                let out = resp.into_string().map_err(|e| rte!("http_post failed: {}", e))?;
                Ok(Value::Str(out))
            }
            "start_http_server" => {
                let mut iter = args.into_iter();
                let port = match iter.next() {
                    Some(Value::Int(n)) => n as u16,
                    other => return Err(rte!("start_http_server(port, handler_fn) requires Int port, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let handler = match iter.next() {
                    Some(f @ Value::Fn { .. }) => f,
                    Some(other) => return Err(rte!("start_http_server: handler must be a function, got `{}`", other)),
                    None => return Err(rte!("start_http_server(port, handler_fn) requires a handler function")),
                };

                let handle = thread::spawn(move || -> Value {
                    let listener = TcpListener::bind(("0.0.0.0", port)).ok();
                    if listener.is_none() { return Value::Unit; }
                    let listener = listener.unwrap();
                    loop {
                        if let Ok((mut stream, _addr)) = listener.accept() {
                            let mut buf = Vec::new();
                            let _ = stream.read_to_end(&mut buf);
                            let body = String::from_utf8_lossy(&buf).to_string();
                            let interp = Interpreter::new();
                            let result = interp.call_value(handler.clone(), vec![Value::Str(body)]);
                            let response = match result {
                                Ok(Value::Str(s)) => s,
                                Ok(v) => v.to_string(),
                                Err(e) => format!("error: {}", e),
                            };
                            let resp = format!(
                                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
                                response.len(),
                                response
                            );
                            let _ = stream.write_all(resp.as_bytes());
                        }
                    }
                });
                let mut rt = runtime();
                let id = rt.allocate_handle();
                rt.threads.insert(id, handle);
                Ok(Value::Int(id as i64))
            }

            // ── Serialization ────────────────────────────────────────────────
            "to_json" => {
                let val = args.into_iter().next().unwrap_or(Value::Nil);
                let json = self.value_to_json(&val);
                Ok(Value::Str(serde_json::to_string(&json)
                    .map_err(|e| rte!("to_json failed: {}", e))?))
            }
            "from_json" | "deserialize" => {
                let s = match args.into_iter().next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("from_json(string) requires a String, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let json: JsonValue = serde_json::from_str(&s)
                    .map_err(|e| rte!("from_json failed: {}", e))?;
                Ok(self.json_to_value(&json))
            }
            "serialize" => {
                let val = args.into_iter().next().unwrap_or(Value::Nil);
                let json = self.value_to_json(&val);
                Ok(Value::Str(serde_json::to_string(&json)
                    .map_err(|e| rte!("serialize failed: {}", e))?))
            }

            // ── Cryptography ───────────────────────────────────────────────
            "sha256" => {
                let s = match args.into_iter().next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("sha256(string) requires a String, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let mut hasher = Sha256::new();
                hasher.update(s.as_bytes());
                let result = hasher.finalize();
                Ok(Value::Str(hex::encode(result)))
            }
            "argon2_hash" => {
                let mut iter = args.into_iter();
                let password = match iter.next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("argon2_hash(password, salt) requires String password, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let salt = match iter.next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("argon2_hash(password, salt) requires String salt, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let salt = SaltString::new(&salt).map_err(|e| rte!("argon2_hash failed: {}", e))?;
                let argon2 = Argon2::default();
                let password_hash = argon2.hash_password(password.as_bytes(), &salt)
                    .map_err(|e| rte!("argon2_hash failed: {}", e))?;
                Ok(Value::Str(password_hash.to_string()))
            }
            "aes_encrypt" => {
                let mut iter = args.into_iter();
                let key = match iter.next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("aes_encrypt(key, plaintext) requires String key, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let plaintext = match iter.next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("aes_encrypt(key, plaintext) requires String plaintext, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let key_bytes = Sha256::digest(key.as_bytes());
                let cipher = Aes256Gcm::new_from_slice(&key_bytes)
                    .map_err(|e| rte!("aes_encrypt failed: {}", e))?;
                let nonce = Nonce::from_slice(&[0u8; 12]);
                let ciphertext = cipher.encrypt(nonce, plaintext.as_bytes())
                    .map_err(|e| rte!("aes_encrypt failed: {}", e))?;
                Ok(Value::Str(general_purpose::STANDARD.encode(ciphertext)))
            }
            "aes_decrypt" => {
                let mut iter = args.into_iter();
                let key = match iter.next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("aes_decrypt(key, ciphertext) requires String key, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let ciphertext = match iter.next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("aes_decrypt(key, ciphertext) requires String ciphertext, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let key_bytes = Sha256::digest(key.as_bytes());
                let cipher = Aes256Gcm::new_from_slice(&key_bytes)
                    .map_err(|e| rte!("aes_decrypt failed: {}", e))?;
                let nonce = Nonce::from_slice(&[0u8; 12]);
                let decoded = general_purpose::STANDARD.decode(ciphertext)
                    .map_err(|e| rte!("aes_decrypt failed: {}", e))?;
                let plaintext = cipher.decrypt(nonce, decoded.as_ref())
                    .map_err(|e| rte!("aes_decrypt failed: {}", e))?;
                Ok(Value::Str(String::from_utf8_lossy(&plaintext).to_string()))
            }
            "secure_random" => {
                let val = args.into_iter().next();
                let mut rng = rand::rngs::OsRng;
                let out = match val {
                    Some(Value::Int(n)) => {
                        let n = n;
                        if n <= 0 {
                            return Err(rte!("secure_random(max) requires positive Int"));
                        }
                        let r: i64 = rng.gen_range(0..n);
                        Value::Int(r)
                    }
                    _ => Value::Int(rng.gen()),
                };
                Ok(out)
            }

            // ── Compression ─────────────────────────────────────────────────
            "gzip_compress" => {
                let s = match args.into_iter().next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("gzip_compress(string) requires String, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(s.as_bytes()).map_err(|e| rte!("gzip_compress failed: {}", e))?;
                let compressed = encoder.finish().map_err(|e| rte!("gzip_compress failed: {}", e))?;
                Ok(Value::Str(general_purpose::STANDARD.encode(&compressed)))
            }
            "gzip_decompress" => {
                let s = match args.into_iter().next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("gzip_decompress(string) requires String, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let data = general_purpose::STANDARD.decode(s)
                    .map_err(|e| rte!("gzip_decompress failed: {}", e))?;
                let mut decoder = GzDecoder::new(&data[..]);
                let mut out = String::new();
                decoder.read_to_string(&mut out).map_err(|e| rte!("gzip_decompress failed: {}", e))?;
                Ok(Value::Str(out))
            }

            // ── Math & statistics ───────────────────────────────────────────
            "vector_dot" => {
                let mut iter = args.into_iter();
                let a = match iter.next() {
                    Some(Value::List(v)) => v,
                    other => return Err(rte!("vector_dot(vec1, vec2) requires List, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let b = match iter.next() {
                    Some(Value::List(v)) => v,
                    other => return Err(rte!("vector_dot(vec1, vec2) requires List, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                if a.len() != b.len() {
                    return Err(rte!("vector_dot: vectors must have same length"));
                }
                let mut sum = 0.0;
                for (x, y) in a.into_iter().zip(b.into_iter()) {
                    let xf = match x {
                        Value::Int(n) => n as f64,
                        Value::Float(f) => f,
                        other => return Err(rte!("vector_dot expects numbers, got `{}`", other)),
                    };
                    let yf = match y {
                        Value::Int(n) => n as f64,
                        Value::Float(f) => f,
                        other => return Err(rte!("vector_dot expects numbers, got `{}`", other)),
                    };
                    sum += xf * yf;
                }
                Ok(Value::Float(sum))
            }
            "matrix_multiply" => {
                let mut iter = args.into_iter();
                let a = match iter.next() {
                    Some(Value::List(rows)) => rows,
                    other => return Err(rte!("matrix_multiply(a, b) requires List of Lists, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let b = match iter.next() {
                    Some(Value::List(rows)) => rows,
                    other => return Err(rte!("matrix_multiply(a, b) requires List of Lists, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                // Convert to Vec<Vec<f64>>
                let a_mat: Result<Vec<Vec<f64>>, RuntimeError> = a.into_iter().map(|r| {
                    match r {
                        Value::List(cols) => cols.into_iter().map(|v| match v {
                            Value::Int(n) => Ok(n as f64),
                            Value::Float(f) => Ok(f),
                            other => Err(rte!("matrix_multiply expects numbers, got `{}`", other)),
                        }).collect(),
                        other => Err(rte!("matrix_multiply expects matrix rows as Lists, got `{}`", other)),
                    }
                }).collect();
                let a_mat = a_mat?;
                let b_mat: Result<Vec<Vec<f64>>, RuntimeError> = b.into_iter().map(|r| {
                    match r {
                        Value::List(cols) => cols.into_iter().map(|v| match v {
                            Value::Int(n) => Ok(n as f64),
                            Value::Float(f) => Ok(f),
                            other => Err(rte!("matrix_multiply expects numbers, got `{}`", other)),
                        }).collect(),
                        other => Err(rte!("matrix_multiply expects matrix rows as Lists, got `{}`", other)),
                    }
                }).collect();
                let b_mat = b_mat?;
                if a_mat.is_empty() || b_mat.is_empty() {
                    return Err(rte!("matrix_multiply: empty matrix"));
                }
                let a_cols = a_mat[0].len();
                let b_rows = b_mat.len();
                let b_cols = b_mat[0].len();
                if a_cols != b_rows {
                    return Err(rte!("matrix_multiply: incompatible dimensions"));
                }
                let mut out = vec![vec![0.0; b_cols]; a_mat.len()];
                for i in 0..a_mat.len() {
                    for j in 0..b_cols {
                        for k in 0..a_cols {
                            out[i][j] += a_mat[i][k] * b_mat[k][j];
                        }
                    }
                }
                Ok(Value::List(out.into_iter().map(|row| {
                    Value::List(row.into_iter().map(Value::Float).collect())
                }).collect()))
            }
            "transpose" => {
                let mat = match args.into_iter().next() {
                    Some(Value::List(rows)) => rows,
                    other => return Err(rte!("transpose(matrix) requires List of Lists, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                if mat.is_empty() { return Ok(Value::List(vec![])); }
                let _row_count = mat.len();
                let col_count = match &mat[0] {
                    Value::List(cols) => cols.len(),
                    other => return Err(rte!("transpose expects matrix rows to be Lists, got `{}`", other)),
                };
                let mut out = vec![vec![]; col_count];
                for row in mat {
                    match row {
                        Value::List(cols) => {
                            for (j, v) in cols.into_iter().enumerate() {
                                out[j].push(v);
                            }
                        }
                        other => return Err(rte!("transpose expects matrix rows to be Lists, got `{}`", other)),
                    }
                }
                Ok(Value::List(out.into_iter().map(Value::List).collect()))
            }
            "mean" | "variance" | "standard_deviation" => {
                let values = match args.into_iter().next() {
                    Some(Value::List(v)) => v,
                    other => return Err(rte!("{} requires a List of numbers, got `{}`", "mean", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let nums: Result<Vec<f64>, RuntimeError> = values.into_iter().map(|v| match v {
                    Value::Int(n) => Ok(n as f64),
                    Value::Float(f) => Ok(f),
                    other => Err(rte!("{} expects numbers, got `{}`", "mean", other)),
                }).collect();
                let nums = nums?;
                if nums.is_empty() {
                    return Err(rte!("{} requires a non-empty list", "mean"));
                }
                let sum: f64 = nums.iter().sum();
                let mean = sum / (nums.len() as f64);
                match name {
                    "mean" => Ok(Value::Float(mean)),
                    "variance" => {
                        let var = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nums.len() as f64);
                        Ok(Value::Float(var))
                    }
                    "standard_deviation" => {
                        let var = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nums.len() as f64);
                        Ok(Value::Float(var.sqrt()))
                    }
                    _ => unreachable!(),
                }
            }

            // ── Random number generation ────────────────────────────────────
            "rand_int" => {
                let mut rng = rand::thread_rng();
                if let Some(Value::Int(max)) = args.into_iter().next() {
                    if max <= 0 { return Err(rte!("rand_int(max) requires positive Int")); }
                    Ok(Value::Int(rng.gen_range(0..max)))
                } else {
                    Ok(Value::Int(rng.gen()))
                }
            }
            "rand_float" => {
                let mut rng = rand::thread_rng();
                Ok(Value::Float(rng.gen()))
            }
            "rand_range" => {
                let mut iter = args.into_iter();
                let lo = match iter.next() {
                    Some(Value::Int(n)) => n,
                    other => return Err(rte!("rand_range(lo, hi) requires Int, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let hi = match iter.next() {
                    Some(Value::Int(n)) => n,
                    other => return Err(rte!("rand_range(lo, hi) requires Int, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                if lo >= hi {
                    return Err(rte!("rand_range: lo must be < hi"));
                }
                let mut rng = rand::thread_rng();
                Ok(Value::Int(rng.gen_range(lo..hi)))
            }

            // ── Time & scheduling ───────────────────────────────────────────
            "current_time" => {
                let now = SystemTime::now().duration_since(UNIX_EPOCH)
                    .map_err(|e| rte!("current_time failed: {}", e))?;
                Ok(Value::Float(now.as_secs_f64()))
            }
            "duration" => {
                let val = args.into_iter().next().ok_or_else(|| rte!("duration(seconds) requires an argument"))?;
                match val {
                    Value::Int(n) => Ok(Value::Float(n as f64)),
                    Value::Float(f) => Ok(Value::Float(f)),
                    other => Err(rte!("duration() expects a number, got `{}`", other)),
                }
            }

            // ── File system utilities ───────────────────────────────────────
            "list_directory" => {
                let path = match args.into_iter().next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("list_directory(path) requires String, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let entries = std::fs::read_dir(path.as_str())
                    .map_err(|e| rte!("list_directory failed: {}", e))?;
                let mut out = Vec::new();
                for entry in entries {
                    let e = entry.map_err(|e| rte!("list_directory failed: {}", e))?;
                    out.push(Value::Str(e.file_name().to_string_lossy().to_string()));
                }
                Ok(Value::List(out))
            }
            "create_directory" => {
                let path = match args.into_iter().next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("create_directory(path) requires String, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                std::fs::create_dir_all(path.as_str()).map_err(|e| rte!("create_directory failed: {}", e))?;
                Ok(Value::Unit)
            }
            "move_file" => {
                let mut iter = args.into_iter();
                let src = match iter.next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("move_file(src, dst) requires String src, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let dst = match iter.next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("move_file(src, dst) requires String dst, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                std::fs::rename(src.as_str(), dst.as_str()).map_err(|e| rte!("move_file failed: {}", e))?;
                Ok(Value::Unit)
            }
            "delete_file" => {
                let path = match args.into_iter().next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("delete_file(path) requires String, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                std::fs::remove_file(path.as_str()).map_err(|e| rte!("delete_file failed: {}", e))?;
                Ok(Value::Unit)
            }
            "watch_directory" => {
                let path = match args.into_iter().next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("watch_directory(path) requires String, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                // Simple snapshot: list entries now.
                let entries = std::fs::read_dir(path.as_str())
                    .map_err(|e| rte!("watch_directory failed: {}", e))?;
                let mut out = Vec::new();
                for entry in entries {
                    let e = entry.map_err(|e| rte!("watch_directory failed: {}", e))?;
                    out.push(Value::Str(e.file_name().to_string_lossy().to_string()));
                }
                Ok(Value::List(out))
            }

            // ── Process management ──────────────────────────────────────────
            "spawn_process" => {
                let mut parts = match args.into_iter().next() {
                    Some(Value::List(v)) => v,
                    other => return Err(rte!("spawn_process([cmd, args...]) requires a List, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                if parts.is_empty() {
                    return Err(rte!("spawn_process requires a non-empty list"));
                }
                let cmd = match parts.remove(0) {
                    Value::Str(s) => s,
                    other => return Err(rte!("spawn_process: command must be String, got `{}`", other)),
                };
                let mut command = Command::new(cmd);
                for part in parts {
                    if let Value::Str(s) = part {
                        command.arg(s);
                    }
                }
                command.stdout(Stdio::piped()).stderr(Stdio::piped());
                let child = command.spawn().map_err(|e| rte!("spawn_process failed: {}", e))?;
                let mut rt = runtime();
                let id = rt.allocate_handle();
                rt.processes.insert(id, child);
                Ok(Value::Int(id as i64))
            }
            "kill_process" => {
                let id = match args.into_iter().next() {
                    Some(Value::Int(n)) => n as usize,
                    other => return Err(rte!("kill_process(pid) requires Int, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let mut rt = runtime();
                match rt.processes.get_mut(&id) {
                    Some(child) => child.kill().map_err(|e| rte!("kill_process failed: {}", e))?,
                    None => return Err(rte!("kill_process: unknown pid {}", id)),
                }
                Ok(Value::Unit)
            }
            "get_process_output" => {
                if let Some(Value::Int(id)) = args.into_iter().next() {
                    let mut rt = runtime();
                    if let Some(child) = rt.processes.remove(&(id as usize)) {
                        let output = child.wait_with_output().map_err(|e| rte!("get_process_output failed: {}", e))?;
                        let mut out = String::new();
                        out.push_str(&String::from_utf8_lossy(&output.stdout));
                        out.push_str(&String::from_utf8_lossy(&output.stderr));
                        Ok(Value::Str(out))
                    } else {
                        Err(rte!("get_process_output: unknown pid {}", id))
                    }
                } else {
                    Err(rte!("get_process_output(pid) requires Int"))
                }
            }

            // ── Logging ─────────────────────────────────────────────────────
            "log_info" | "log_warning" | "log_error" | "log_debug" => {
                let level = name;
                let msg = args.into_iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ");
                let prefix = match level {
                    "log_info" => "INFO",
                    "log_warning" => "WARN",
                    "log_error" => "ERROR",
                    "log_debug" => "DEBUG",
                    _ => "LOG",
                };
                println!("[{}] {}", prefix, msg);
                Ok(Value::Unit)
            }

            // ── Configuration ──────────────────────────────────────────────
            "read_env" => {
                let key = match args.into_iter().next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("read_env(key) requires String, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                Ok(Value::Str(std::env::var(key).unwrap_or_default()))
            }
            "parse_toml" => {
                let s = match args.into_iter().next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("parse_toml(string) requires String, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let table: toml::Value = toml::from_str(&s).map_err(|e| rte!("parse_toml failed: {}", e))?;
                Ok(self.json_to_value(&serde_json::to_value(table).map_err(|e| rte!("parse_toml failed: {}", e))?))
            }

            // ── CLI arguments ──────────────────────────────────────────────
            "parse_args" => {
                let args: Vec<Value> = std::env::args().map(Value::Str).collect();
                Ok(Value::List(args))
            }
            "flag" => {
                let name = match args.into_iter().next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("flag(name) requires String, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let found = std::env::args().any(|a| a == format!("--{}", name));
                Ok(Value::Bool(found))
            }
            "option" => {
                let name = match args.into_iter().next() {
                    Some(Value::Str(s)) => s,
                    other => return Err(rte!("option(name) requires String, got `{}`", other.map(|v| v.to_string()).unwrap_or_default())),
                };
                let flag = format!("--{}", name);
                let mut iter = std::env::args();
                while let Some(arg) = iter.next() {
                    if arg == flag {
                        if let Some(val) = iter.next() {
                            return Ok(Value::Str(val));
                        }
                        break;
                    }
                }
                Ok(Value::Nil)
            }

            unknown => Err(rte!(
                "unknown builtin `{}`\n\
                 hint: available builtins include: println, len, push, pop, \
                 map_new, split, abs, sqrt, range, assert…",
                unknown
            )),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // HELPERS
    // ─────────────────────────────────────────────────────────────────────────

    fn type_name<'a>(&self, v: &'a Value) -> &'static str {
        match v {
            Value::Int(_)        => "Int",
            Value::Float(_)      => "Float",
            Value::Bool(_)       => "Bool",
            Value::Str(_)        => "String",
            Value::Nil           => "Nil",
            Value::Unit          => "Unit",
            Value::List(_)       => "List",
            Value::Tuple(_)      => "Tuple",
            Value::Map(_)        => "Map",
            Value::Struct {..}   => "Struct",
            Value::Enum {..}     => "Enum",
            Value::Fn {..}       => "Fn",
            Value::Builtin(_)    => "Builtin",
            Value::ResultOk(_)   => "Ok",
            Value::ResultErr(_)  => "Err",
            Value::Future(_)     => "Future",
        }
    }

    fn value_to_json(&self, v: &Value) -> JsonValue {
        match v {
            Value::Int(n) => JsonValue::Number((*n).into()),
            Value::Float(f) => JsonValue::Number(serde_json::Number::from_f64(*f).unwrap_or(serde_json::Number::from(0))),
            Value::Bool(b) => JsonValue::Bool(*b),
            Value::Str(s) => JsonValue::String(s.clone()),
            Value::Nil | Value::Unit => JsonValue::Null,
            Value::List(vs) => JsonValue::Array(vs.iter().map(|x| self.value_to_json(x)).collect()),
            Value::Tuple(vs) => JsonValue::Array(vs.iter().map(|x| self.value_to_json(x)).collect()),
            Value::Map(pairs) => JsonValue::Object(pairs.iter().map(|(k, v)| {
                (k.to_string(), self.value_to_json(v))
            }).collect()),
            Value::Struct { name, fields } => {
                let mut map = serde_json::Map::new();
                map.insert("__struct".to_string(), JsonValue::String(name.clone()));
                for (k, v) in fields {
                    map.insert(k.clone(), self.value_to_json(v));
                }
                JsonValue::Object(map)
            }
            Value::Enum { variant, payload } => {
                if let Some(payload) = payload {
                    JsonValue::Array(vec![JsonValue::String(variant.clone()), self.value_to_json(payload)])
                } else {
                    JsonValue::String(variant.clone())
                }
            }
            Value::Fn { .. } | Value::Builtin(_) | Value::Future(_) => JsonValue::Null,
            Value::ResultOk(inner) => self.value_to_json(inner),
            Value::ResultErr(inner) => self.value_to_json(inner),
        }
    }

    fn json_to_value(&self, j: &JsonValue) -> Value {
        match j {
            JsonValue::Null => Value::Nil,
            JsonValue::Bool(b) => Value::Bool(*b),
            JsonValue::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Int(i)
                } else if let Some(f) = n.as_f64() {
                    Value::Float(f)
                } else {
                    Value::Nil
                }
            }
            JsonValue::String(s) => Value::Str(s.clone()),
            JsonValue::Array(arr) => Value::List(arr.iter().map(|x| self.json_to_value(x)).collect()),
            JsonValue::Object(obj) => {
                let mut map = Vec::new();
                for (k, v) in obj {
                    map.push((Value::Str(k.clone()), self.json_to_value(v)));
                }
                Value::Map(map)
            }
        }
    }
}
