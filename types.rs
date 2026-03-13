// ─────────────────────────────────────────────────────────────────────────────
// TYPE SYSTEM  &  SEMANTIC ANALYSIS
// ─────────────────────────────────────────────────────────────────────────────
#![allow(dead_code)]
use std::collections::HashMap;
use crate::ast::*;
use crate::lexer::Span;

// ── Runtime / resolved types ──────────────────────────────────────────────────

/// Fully resolved concrete types after inference.
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Float,
    Bool,
    Str,
    Unit,
    Nil,

    List(Box<Type>),
    Map(Box<Type>, Box<Type>),
    Tuple(Vec<Type>),

    // User-defined
    Struct(String),
    Enum(String),

    // Callable
    Fn { params: Vec<Type>, ret: Box<Type> },

    // Async wrapper
    Future(Box<Type>),

    // Type variable (for inference, filled in later)
    Var(u32),
    Unknown,
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Int    => write!(f, "Int"),
            Type::Float  => write!(f, "Float"),
            Type::Bool   => write!(f, "Bool"),
            Type::Str    => write!(f, "String"),
            Type::Unit   => write!(f, "()"),
            Type::Nil    => write!(f, "Nil"),
            Type::List(t)      => write!(f, "List<{}>", t),
            Type::Map(k, v)    => write!(f, "Map<{}, {}>", k, v),
            Type::Tuple(ts)    => {
                write!(f, "(")?;
                for (i, t) in ts.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", t)?;
                }
                write!(f, ")")
            }
            Type::Struct(n)    => write!(f, "{}", n),
            Type::Enum(n)      => write!(f, "{}", n),
            Type::Fn{params,ret} => {
                write!(f, "fn(")?;
                for (i,p) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            Type::Future(t)    => write!(f, "Future<{}>", t),
            Type::Var(id)      => write!(f, "?T{}", id),
            Type::Unknown      => write!(f, "?"),
        }
    }
}

// ── Symbol table (scoped) ─────────────────────────────────────────────────────

/// One scope level — function, block, module, etc.
#[derive(Debug, Default)]
pub struct Scope {
    pub symbols: HashMap<String, Symbol>,
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name:    String,
    pub ty:      Type,
    pub mutable: bool,
    pub defined_at: Span,
}

/// A stack of scopes. Inner-most scope is last.
pub struct SymbolTable {
    scopes:   Vec<Scope>,
    next_var: u32,
}

impl SymbolTable {
    pub fn new() -> Self { Self { scopes: vec![Scope::default()], next_var: 0 } }

    pub fn push_scope(&mut self) { self.scopes.push(Scope::default()); }

    pub fn pop_scope(&mut self) { self.scopes.pop(); }

    pub fn define(&mut self, sym: Symbol) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.symbols.insert(sym.name.clone(), sym);
        }
    }

    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        for scope in self.scopes.iter().rev() {
            if let Some(s) = scope.symbols.get(name) { return Some(s); }
        }
        None
    }

    /// Create a fresh type variable for inference.
    pub fn fresh_var(&mut self) -> Type { let id = self.next_var; self.next_var += 1; Type::Var(id) }
}

// ── Semantic error ────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct SemanticError {
    pub msg:  String,
    pub span: Span,
    pub hint: Option<String>,
}

impl std::fmt::Display for SemanticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "error[S001]: {} at {}:{}:{}",
            self.msg, self.span.file, self.span.line, self.span.col)?;
        if let Some(h) = &self.hint { write!(f, "\n  hint: {}", h)?; }
        Ok(())
    }
}

// ── Analyser ─────────────────────────────────────────────────────────────────

pub struct Analyser {
    pub symbols: SymbolTable,
    pub errors:  Vec<SemanticError>,
    // Struct / enum registries
    struct_defs: HashMap<String, StructDef>,
    enum_defs:   HashMap<String, EnumDef>,
    fn_sigs:     HashMap<String, (Vec<Type>, Type)>,
}

impl Analyser {
    pub fn new() -> Self {
        Self {
            symbols:     SymbolTable::new(),
            errors:      Vec::new(),
            struct_defs: HashMap::new(),
            enum_defs:   HashMap::new(),
            fn_sigs:     HashMap::new(),
        }
    }

    fn err(&mut self, msg: &str, span: Span, hint: Option<&str>) {
        self.errors.push(SemanticError {
            msg:  msg.to_string(),
            span,
            hint: hint.map(|s| s.to_string()),
        });
    }

    // ── Module-level first pass: register all top-level names ─────────────────

    pub fn analyse_module(&mut self, module: &Module) {
        // 1. Register structs and enums
        for s in &module.structs { self.struct_defs.insert(s.name.clone(), s.clone()); }
        for e in &module.enums   { self.enum_defs.insert(e.name.clone(), e.clone()); }

        // 2. Pre-register function signatures (allows mutual recursion)
        for f in &module.fns {
            let params: Vec<Type> = f.params.iter()
                .map(|p| p.ty.as_ref().map(|t| self.lower_type(t)).unwrap_or(Type::Unknown))
                .collect();
            let ret = f.ret_ty.as_ref().map(|t| self.lower_type(t)).unwrap_or(Type::Unit);
            self.fn_sigs.insert(f.name.clone(), (params.clone(), ret.clone()));
            self.symbols.define(Symbol {
                name:       f.name.clone(),
                ty:         Type::Fn { params, ret: Box::new(ret) },
                mutable:    false,
                defined_at: f.span.clone(),
            });
        }

        // 3. Check function bodies
        for f in &module.fns { self.analyse_fn(f); }

        // 4. Top-level statements
        for stmt in &module.stmts { self.analyse_stmt(stmt); }
    }

    // ── Convert written TypeExpr → resolved Type ──────────────────────────────

    fn lower_type(&mut self, ty: &TypeExpr) -> Type {
        match ty {
            TypeExpr::Named(n) => match n.as_str() {
                "Int"    => Type::Int,
                "Float"  => Type::Float,
                "Bool"   => Type::Bool,
                "String" => Type::Str,
                "()"     => Type::Unit,
                other    => {
                    if self.struct_defs.contains_key(other) { return Type::Struct(other.to_string()); }
                    if self.enum_defs.contains_key(other)   { return Type::Enum(other.to_string()); }
                    Type::Unknown
                }
            },
            TypeExpr::Generic(n, args) => match n.as_str() {
                "List" if args.len() == 1 =>
                    Type::List(Box::new(self.lower_type(&args[0]))),
                "Map"  if args.len() == 2 =>
                    Type::Map(Box::new(self.lower_type(&args[0])), Box::new(self.lower_type(&args[1]))),
                _ => Type::Unknown,
            },
            TypeExpr::Tuple(ts) =>
                Type::Tuple(ts.iter().map(|t| self.lower_type(t)).collect()),
            TypeExpr::Fn(params, ret) =>
                Type::Fn {
                    params: params.iter().map(|t| self.lower_type(t)).collect(),
                    ret:    Box::new(self.lower_type(ret)),
                },
            TypeExpr::Optional(inner) =>
                // Represent Optional<T> as Enum("Option") placeholder
                Type::Enum(format!("Option<{}>", self.lower_type(inner))),
            TypeExpr::Infer => self.symbols.fresh_var(),
        }
    }

    // ── Function analysis ─────────────────────────────────────────────────────

    fn analyse_fn(&mut self, f: &FnDef) {
        self.symbols.push_scope();

        // Bind parameters into new scope
        for p in &f.params {
            let ty = p.ty.as_ref().map(|t| self.lower_type(t)).unwrap_or_else(|| self.symbols.fresh_var());
            self.symbols.define(Symbol {
                name: p.name.clone(), ty, mutable: false, defined_at: p.span.clone(),
            });
        }

        for stmt in &f.body { self.analyse_stmt(stmt); }
        self.symbols.pop_scope();
    }

    // ── Statement analysis ────────────────────────────────────────────────────

    fn analyse_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let { mutable, name, ty, value, span } => {
                let val_ty = self.analyse_expr(value);
                let declared_ty = ty.as_ref().map(|t| self.lower_type(t));
                let final_ty = if let Some(dt) = declared_ty {
                    if !self.types_compatible(&dt, &val_ty) {
                        self.err(
                            &format!("type mismatch: declared `{}` but got `{}`", dt, val_ty),
                            span.clone(),
                            Some("check that the value matches the declared type"),
                        );
                    }
                    dt
                } else { val_ty };

                self.symbols.define(Symbol {
                    name: name.clone(), ty: final_ty, mutable: *mutable,
                    defined_at: span.clone(),
                });
            }
            Stmt::Return(val, span) => {
                if let Some(v) = val { let _ = self.analyse_expr(v); }
                let _ = span;
            }
            Stmt::Expr(e) => { self.analyse_expr(e); }
            Stmt::While { cond, body, span } => {
                let cty = self.analyse_expr(cond);
                if cty != Type::Bool && cty != Type::Unknown {
                    self.err("while condition must be `Bool`", span.clone(),
                        Some("change condition to return a Bool"));
                }
                self.symbols.push_scope();
                for s in body { self.analyse_stmt(s); }
                self.symbols.pop_scope();
            }
            Stmt::For { var, iter, body, span } => {
                let iter_ty = self.analyse_expr(iter);
                let elem_ty = match &iter_ty {
                    Type::List(inner) => *inner.clone(),
                    _ => Type::Unknown,
                };
                self.symbols.push_scope();
                self.symbols.define(Symbol {
                    name: var.clone(), ty: elem_ty, mutable: false,
                    defined_at: span.clone(),
                });
                for s in body { self.analyse_stmt(s); }
                self.symbols.pop_scope();
            }
            Stmt::Import { path, .. } => {
                // TODO: resolve module path and import symbols
                let _ = path;
            }
        }
    }

    // ── Expression type inference ─────────────────────────────────────────────

    pub fn analyse_expr(&mut self, expr: &Expr) -> Type {
        match expr {
            Expr::Int(_, _)   => Type::Int,
            Expr::Float(_, _) => Type::Float,
            Expr::Bool(_, _)  => Type::Bool,
            Expr::Str(_, _)   => Type::Str,
            Expr::Nil(_)      => Type::Nil,

            Expr::Var(name, span) => {
                match self.symbols.lookup(name) {
                    Some(sym) => sym.ty.clone(),
                    None => {
                        self.err(
                            &format!("variable `{}` used before declaration", name),
                            span.clone(),
                            Some("declare it with `let` before using it"),
                        );
                        Type::Unknown
                    }
                }
            }

            Expr::Unary { op, expr, .. } => {
                let ty = self.analyse_expr(expr);
                match op {
                    UnaryOp::Neg => if ty == Type::Int || ty == Type::Float { ty } else { Type::Unknown },
                    UnaryOp::Not => if ty == Type::Bool { Type::Bool } else { Type::Unknown },
                }
            }

            Expr::Binary { op, left, right, span } => {
                let lt = self.analyse_expr(left);
                let rt = self.analyse_expr(right);
                match op {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul
                    | BinaryOp::Div | BinaryOp::Mod => {
                        if lt == Type::Int && rt == Type::Int { Type::Int }
                        else if (lt == Type::Float || lt == Type::Int)
                             && (rt == Type::Float || rt == Type::Int) { Type::Float }
                        else if *op == BinaryOp::Add && lt == Type::Str && rt == Type::Str {
                            Type::Str   // string concatenation
                        } else {
                            self.err(&format!("operator `{:?}` not valid for `{}` and `{}`",
                                op, lt, rt), span.clone(), None);
                            Type::Unknown
                        }
                    }
                    BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt
                    | BinaryOp::Gt | BinaryOp::LtEq | BinaryOp::GtEq => Type::Bool,
                    BinaryOp::And | BinaryOp::Or => {
                        if lt == Type::Bool && rt == Type::Bool { Type::Bool }
                        else {
                            self.err("logical operators require Bool operands",
                                span.clone(), Some("convert to Bool first"));
                            Type::Unknown
                        }
                    }
                }
            }

            Expr::Assign { target, value, span } => {
                // Check mutability
                if let Expr::Var(name, _) = target.as_ref() {
                    if let Some(sym) = self.symbols.lookup(name) {
                        if !sym.mutable {
                            self.err(&format!("cannot assign to immutable variable `{}`", name),
                                span.clone(), Some("declare with `let mut` to allow mutation"));
                        }
                    }
                }
                self.analyse_expr(value)
            }

            Expr::Call { callee, args, span } => {
                let callee_ty = self.analyse_expr(callee);
                let arg_tys: Vec<Type> = args.iter().map(|a| self.analyse_expr(a)).collect();
                match callee_ty {
                    Type::Fn { params, ret } => {
                        if params.len() != arg_tys.len() {
                            self.err(&format!("expected {} argument(s), got {}",
                                params.len(), arg_tys.len()), span.clone(), None);
                        }
                        *ret
                    }
                    _ => { Type::Unknown }
                }
            }

            Expr::Array(elems, _) => {
                if elems.is_empty() { return Type::List(Box::new(Type::Unknown)); }
                let elem_ty = self.analyse_expr(&elems[0]);
                Type::List(Box::new(elem_ty))
            }

            Expr::Tuple(elems, _) =>
                Type::Tuple(elems.iter().map(|e| self.analyse_expr(e)).collect()),

            Expr::Block(stmts, tail, _) => {
                self.symbols.push_scope();
                for s in stmts { self.analyse_stmt(s); }
                let ty = tail.as_ref().map(|e| self.analyse_expr(e)).unwrap_or(Type::Unit);
                self.symbols.pop_scope();
                ty
            }

            Expr::If { cond, then_branch, else_branch, span } => {
                let cty = self.analyse_expr(cond);
                if cty != Type::Bool && cty != Type::Unknown {
                    self.err("if condition must be Bool", span.clone(), None);
                }
                let then_ty = self.analyse_expr(then_branch);
                if let Some(eb) = else_branch {
                    let else_ty = self.analyse_expr(eb);
                    if !self.types_compatible(&then_ty, &else_ty) {
                        self.err(&format!("if branches have incompatible types: `{}` vs `{}`",
                            then_ty, else_ty), span.clone(),
                            Some("both branches must return the same type"));
                    }
                }
                then_ty
            }

            Expr::Field { object, field, span } => {
                let obj_ty = self.analyse_expr(object);
                if let Type::Struct(name) = &obj_ty {
                    if let Some(def) = self.struct_defs.get(name).cloned() {
                        if let Some((_, fty)) = def.fields.iter().find(|(n, _)| n == field) {
                            return self.lower_type(fty);
                        }
                    }
                    self.err(&format!("no field `{}` on `{}`", field, name), span.clone(), None);
                }
                Type::Unknown
            }

            Expr::Match { subject, arms, .. } => {
                let _ = self.analyse_expr(subject);
                // Return type of first arm (simplified; should unify all arm types)
                arms.first().map(|a| self.analyse_expr(&a.body)).unwrap_or(Type::Unit)
            }

            Expr::Lambda { params, body, .. } => {
                self.symbols.push_scope();
                let param_tys: Vec<Type> = params.iter().map(|p| {
                    let ty = p.ty.as_ref().map(|t| self.lower_type(t))
                        .unwrap_or_else(|| self.symbols.fresh_var());
                    self.symbols.define(Symbol {
                        name: p.name.clone(), ty: ty.clone(), mutable: false,
                        defined_at: p.span.clone(),
                    });
                    ty
                }).collect();
                let ret = self.analyse_expr(body);
                self.symbols.pop_scope();
                Type::Fn { params: param_tys, ret: Box::new(ret) }
            }

            Expr::Await { expr, .. } => {
                let inner = self.analyse_expr(expr);
                match inner {
                    Type::Future(t) => *t,
                    other => other,   // tolerate for now
                }
            }

            _ => Type::Unknown,
        }
    }

    // ── Type unification (simplified) ─────────────────────────────────────────

    fn types_compatible(&self, a: &Type, b: &Type) -> bool {
        if a == b { return true; }
        matches!((a, b), (Type::Unknown, _) | (_, Type::Unknown)
            | (Type::Var(_), _) | (_, Type::Var(_)))
    }
}
