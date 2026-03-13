// ─────────────────────────────────────────────────────────────────────────────
// AST  (Abstract Syntax Tree node definitions)
// ─────────────────────────────────────────────────────────────────────────────
#![allow(dead_code)]
use crate::lexer::Span;

// ── Types ─────────────────────────────────────────────────────────────────────

/// The type annotation syntax a programmer writes.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeExpr {
    Named(String),                          // Int, String, MyStruct
    Generic(String, Vec<TypeExpr>),         // List<Int>, Map<String, Int>
    Fn(Vec<TypeExpr>, Box<TypeExpr>),       // fn(Int, Int) -> Bool
    Tuple(Vec<TypeExpr>),                   // (Int, String)
    Optional(Box<TypeExpr>),                // Int?
    Infer,                                  // _ — let the compiler decide
}

// ── Expressions ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Expr {
    // Literals
    Int(i64,    Span),
    Float(f64,  Span),
    Bool(bool,  Span),
    Str(String, Span),
    Nil(Span),

    // Variable reference
    Var(String, Span),

    // Unary  e.g.  -x  !b
    Unary { op: UnaryOp, expr: Box<Expr>, span: Span },

    // Binary  e.g.  a + b
    Binary { op: BinaryOp, left: Box<Expr>, right: Box<Expr>, span: Span },

    // Assignment  x = expr
    Assign { target: Box<Expr>, value: Box<Expr>, span: Span },

    // Call  foo(a, b)
    Call { callee: Box<Expr>, args: Vec<Expr>, span: Span },

    // Member access  obj.field
    Field { object: Box<Expr>, field: String, span: Span },

    // Indexing  arr[i]
    Index { object: Box<Expr>, index: Box<Expr>, span: Span },

    // Lambda / closure  |a, b| expr
    Lambda { params: Vec<Param>, body: Box<Expr>, span: Span },

    // Block expression  { stmts... ; expr }
    Block(Vec<Stmt>, Option<Box<Expr>>, Span),

    // If expression  if cond { a } else { b }
    If { cond: Box<Expr>, then_branch: Box<Expr>, else_branch: Option<Box<Expr>>, span: Span },

    // Match expression
    Match { subject: Box<Expr>, arms: Vec<MatchArm>, span: Span },

    // Array literal  [1, 2, 3]
    Array(Vec<Expr>, Span),

    // Struct literal  Point { x: 1, y: 2 }
    StructLit { name: String, fields: Vec<(String, Expr)>, span: Span },

    // Tuple literal  (1, "hello")
    Tuple(Vec<Expr>, Span),

    // Range  1..10
    Range { start: Box<Expr>, end: Box<Expr>, inclusive: bool, span: Span },

    // Await  expr.await  (async support)
    Await { expr: Box<Expr>, span: Span },
}

impl Expr {
    pub fn span(&self) -> &Span {
        match self {
            Expr::Int(_, s) | Expr::Float(_, s) | Expr::Bool(_, s)
            | Expr::Str(_, s) | Expr::Nil(s) | Expr::Var(_, s) => s,
            Expr::Unary  { span, .. } | Expr::Binary { span, .. }
            | Expr::Assign { span, .. } | Expr::Call   { span, .. }
            | Expr::Field  { span, .. } | Expr::Index  { span, .. }
            | Expr::Lambda { span, .. } | Expr::Block(_, _, span)
            | Expr::If     { span, .. } | Expr::Match  { span, .. }
            | Expr::Array(_, span)      | Expr::StructLit { span, .. }
            | Expr::Tuple(_, span)      | Expr::Range  { span, .. }
            | Expr::Await  { span, .. } => span,
        }
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Mod,
    Eq, NotEq,
    Lt, Gt, LtEq, GtEq,
    And, Or,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp { Neg, Not }

// ── Patterns  (used in match arms & let destructuring) ───────────────────────

#[derive(Debug, Clone)]
pub enum Pattern {
    Wildcard(Span),                             // _
    Binding(String, Span),                      // name
    Literal(Expr),                              // 42  "hello"  true
    Tuple(Vec<Pattern>, Span),                  // (a, b)
    Struct(String, Vec<(String, Pattern)>, Span),// Point { x, y }
    Enum(String, Option<Box<Pattern>>, Span),   // Some(x) / None
    Or(Vec<Pattern>, Span),                     // pat1 | pat2
    Guard(Box<Pattern>, Box<Expr>, Span),       // pat if cond
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body:    Expr,
}

// ── Statements ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Stmt {
    // let [mut] name [: Type] = expr
    Let {
        mutable: bool,
        name:    String,
        ty:      Option<TypeExpr>,
        value:   Expr,
        span:    Span,
    },
    // Expression statement
    Expr(Expr),
    // return expr
    Return(Option<Expr>, Span),
    // while cond { body }
    While { cond: Expr, body: Vec<Stmt>, span: Span },
    // for x in iter { body }
    For { var: String, iter: Expr, body: Vec<Stmt>, span: Span },
    // import path [as alias]
    Import { path: Vec<String>, alias: Option<String>, span: Span },
}

// ── Top-level items ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty:   Option<TypeExpr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FnDef {
    pub name:    String,
    pub params:  Vec<Param>,
    pub ret_ty:  Option<TypeExpr>,
    pub body:    Vec<Stmt>,
    pub is_pub:  bool,
    pub is_async:bool,
    pub span:    Span,
}

#[derive(Debug, Clone)]
pub struct StructDef {
    pub name:   String,
    pub fields: Vec<(String, TypeExpr)>,
    pub is_pub: bool,
    pub span:   Span,
}

#[derive(Debug, Clone)]
pub struct EnumDef {
    pub name:     String,
    pub variants: Vec<EnumVariant>,
    pub is_pub:   bool,
    pub span:     Span,
}

#[derive(Debug, Clone)]
pub struct EnumVariant {
    pub name:    String,
    pub payload: Option<TypeExpr>,  // Some(Int), None
    pub span:    Span,
}

/// One source file = one Module.
#[derive(Debug, Clone)]
pub struct Module {
    pub name:    String,
    pub fns:     Vec<FnDef>,
    pub structs: Vec<StructDef>,
    pub enums:   Vec<EnumDef>,
    pub stmts:   Vec<Stmt>,          // top-level statements
}
