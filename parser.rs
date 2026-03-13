// ─────────────────────────────────────────────────────────────────────────────
// PARSER  (token stream → AST)
// ─────────────────────────────────────────────────────────────────────────────
use crate::ast::*;
use crate::lexer::{Span, Token, TokenKind};

pub struct Parser {
    tokens:  Vec<Token>,
    pos:     usize,
    errors:  Vec<ParseError>,
}

// ── Error type ────────────────────────────────────────────────────────────────
#[derive(Debug)]
pub struct ParseError {
    pub msg:  String,
    pub span: Span,
    pub hint: Option<String>,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "error[P001]: {} at {}:{}:{}", self.msg,
            self.span.file, self.span.line, self.span.col)?;
        if let Some(h) = &self.hint {
            write!(f, "\n  hint: {}", h)?;
        }
        Ok(())
    }
}

// ── Core parser helpers ───────────────────────────────────────────────────────
impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0, errors: Vec::new() }
    }

    fn peek(&self) -> &TokenKind { &self.tokens[self.pos].kind }
    fn span(&self) -> Span       { self.tokens[self.pos].span.clone() }

    fn advance(&mut self) -> &Token {
        let t = &self.tokens[self.pos];
        if t.kind != TokenKind::Eof { self.pos += 1; }
        t
    }

    fn check(&self, k: &TokenKind) -> bool { self.peek() == k }

    fn eat(&mut self, k: &TokenKind) -> bool {
        if self.peek() == k { self.advance(); true } else { false }
    }

    fn expect(&mut self, k: &TokenKind, hint: &str) -> Result<Span, ParseError> {
        let span = self.span();
        if self.eat(k) { Ok(span) }
        else {
            Err(ParseError {
                msg:  format!("expected `{:?}`, found `{:?}`", k, self.peek()),
                span,
                hint: Some(hint.to_string()),
            })
        }
    }

    fn err(&self, msg: &str, hint: &str) -> ParseError {
        ParseError { msg: msg.to_string(), span: self.span(), hint: Some(hint.to_string()) }
    }

    /// Consume tokens until a synchronisation point (error recovery).
    fn synchronise(&mut self) {
        while !matches!(self.peek(),
            TokenKind::Eof | TokenKind::Fn | TokenKind::Let |
            TokenKind::Return | TokenKind::If | TokenKind::While |
            TokenKind::For | TokenKind::Struct | TokenKind::Enum) {
            self.advance();
        }
    }

    pub fn errors(&self) -> &[ParseError] { &self.errors }
}

// ── Module / top-level ────────────────────────────────────────────────────────
impl Parser {
    pub fn parse_module(&mut self, name: &str) -> Module {
        let mut module = Module {
            name:    name.to_string(),
            fns:     Vec::new(),
            structs: Vec::new(),
            enums:   Vec::new(),
            stmts:   Vec::new(),
        };
        while !self.check(&TokenKind::Eof) {
            let is_pub = self.eat(&TokenKind::Pub);
            match self.peek().clone() {
                TokenKind::Fn => {
                    match self.parse_fn(is_pub) {
                        Ok(f)  => module.fns.push(f),
                        Err(e) => { self.errors.push(e); self.synchronise(); }
                    }
                }
                TokenKind::Struct => {
                    match self.parse_struct(is_pub) {
                        Ok(s)  => module.structs.push(s),
                        Err(e) => { self.errors.push(e); self.synchronise(); }
                    }
                }
                TokenKind::Enum => {
                    match self.parse_enum(is_pub) {
                        Ok(e)  => module.enums.push(e),
                        Err(e) => { self.errors.push(e); self.synchronise(); }
                    }
                }
                _ => {
                    match self.parse_stmt() {
                        Ok(s)  => module.stmts.push(s),
                        Err(e) => { self.errors.push(e); self.synchronise(); }
                    }
                }
            }
        }
        module
    }

    fn parse_fn(&mut self, is_pub: bool) -> Result<FnDef, ParseError> {
        let span = self.span();
        self.expect(&TokenKind::Fn, "start function with `fn`")?;
        let is_async = self.eat(&TokenKind::Async);
        let name = match self.peek().clone() {
            TokenKind::Ident(n) => { self.advance(); n }
            _ => return Err(self.err("expected function name", "provide an identifier after `fn`")),
        };
        self.expect(&TokenKind::LParen, "open parameter list with `(`")?;
        let params = self.parse_params()?;
        self.expect(&TokenKind::RParen, "close parameter list with `)`")?;
        let ret_ty = if self.eat(&TokenKind::Arrow) {
            Some(self.parse_type_expr()?)
        } else { None };
        self.expect(&TokenKind::LBrace, "open function body with `{`")?;
        let body = self.parse_block_body()?;
        self.expect(&TokenKind::RBrace, "close function body with `}`")?;
        Ok(FnDef { name, params, ret_ty, body, is_pub, is_async, span })
    }

    fn parse_params(&mut self) -> Result<Vec<Param>, ParseError> {
        let mut params = Vec::new();
        while !matches!(self.peek(), TokenKind::RParen | TokenKind::Eof) {
            let span = self.span();
            let name = match self.peek().clone() {
                TokenKind::Ident(n) => { self.advance(); n }
                _ => return Err(self.err("expected parameter name", "parameter must be an identifier")),
            };
            let ty = if self.eat(&TokenKind::Colon) {
                Some(self.parse_type_expr()?)
            } else { None };
            params.push(Param { name, ty, span });
            if !self.eat(&TokenKind::Comma) { break; }
        }
        Ok(params)
    }

    fn parse_struct(&mut self, is_pub: bool) -> Result<StructDef, ParseError> {
        let span = self.span();
        self.expect(&TokenKind::Struct, "expected `struct`")?;
        let name = match self.peek().clone() {
            TokenKind::Ident(n) => { self.advance(); n }
            _ => return Err(self.err("expected struct name", "")),
        };
        self.expect(&TokenKind::LBrace, "open struct body with `{`")?;
        let mut fields = Vec::new();
        while !matches!(self.peek(), TokenKind::RBrace | TokenKind::Eof) {
            let fname = match self.peek().clone() {
                TokenKind::Ident(n) => { self.advance(); n }
                _ => return Err(self.err("expected field name", "")),
            };
            self.expect(&TokenKind::Colon, "field name must be followed by `:`")?;
            let fty = self.parse_type_expr()?;
            fields.push((fname, fty));
            self.eat(&TokenKind::Comma);
        }
        self.expect(&TokenKind::RBrace, "close struct with `}`")?;
        Ok(StructDef { name, fields, is_pub, span })
    }

    fn parse_enum(&mut self, is_pub: bool) -> Result<EnumDef, ParseError> {
        let span = self.span();
        self.expect(&TokenKind::Enum, "expected `enum`")?;
        let name = match self.peek().clone() {
            TokenKind::Ident(n) => { self.advance(); n }
            _ => return Err(self.err("expected enum name", "")),
        };
        self.expect(&TokenKind::LBrace, "open enum body with `{`")?;
        let mut variants = Vec::new();
        while !matches!(self.peek(), TokenKind::RBrace | TokenKind::Eof) {
            let vspan = self.span();
            let vname = match self.peek().clone() {
                TokenKind::Ident(n) => { self.advance(); n }
                _ => return Err(self.err("expected variant name", "")),
            };
            let payload = if self.eat(&TokenKind::LParen) {
                let t = self.parse_type_expr()?;
                self.expect(&TokenKind::RParen, "close variant payload with `)`")?;
                Some(t)
            } else { None };
            variants.push(EnumVariant { name: vname, payload, span: vspan });
            self.eat(&TokenKind::Comma);
        }
        self.expect(&TokenKind::RBrace, "close enum with `}`")?;
        Ok(EnumDef { name, variants, is_pub, span })
    }
}

// ── Statements ────────────────────────────────────────────────────────────────
impl Parser {
    fn parse_block_body(&mut self) -> Result<Vec<Stmt>, ParseError> {
        let mut stmts = Vec::new();
        while !matches!(self.peek(), TokenKind::RBrace | TokenKind::Eof) {
            stmts.push(self.parse_stmt()?);
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        let span = self.span();
        match self.peek().clone() {
            TokenKind::Let => {
                self.advance();
                let mutable = self.eat(&TokenKind::Ident("mut".to_string()));
                let name = match self.peek().clone() {
                    TokenKind::Ident(n) => { self.advance(); n }
                    _ => return Err(self.err("expected variable name after `let`",
                        "write: let x = value")),
                };
                let ty = if self.eat(&TokenKind::Colon) {
                    Some(self.parse_type_expr()?)
                } else { None };
                self.expect(&TokenKind::Assign, "missing `=` in let binding")?;
                let value = self.parse_expr()?;
                self.eat(&TokenKind::Semicolon);
                Ok(Stmt::Let { mutable, name, ty, value, span })
            }
            TokenKind::Return => {
                self.advance();
                let val = if !matches!(self.peek(), TokenKind::Semicolon | TokenKind::RBrace) {
                    Some(self.parse_expr()?)
                } else { None };
                self.eat(&TokenKind::Semicolon);
                Ok(Stmt::Return(val, span))
            }
            TokenKind::While => {
                self.advance();
                let cond = self.parse_expr()?;
                self.expect(&TokenKind::LBrace, "open while body with `{`")?;
                let body = self.parse_block_body()?;
                self.expect(&TokenKind::RBrace, "close while body with `}`")?;
                Ok(Stmt::While { cond, body, span })
            }
            TokenKind::For => {
                self.advance();
                let var = match self.peek().clone() {
                    TokenKind::Ident(n) => { self.advance(); n }
                    _ => return Err(self.err("expected variable in for loop", "write: for x in ...")),
                };
                self.expect(&TokenKind::In, "expected `in` keyword")?;
                let iter = self.parse_expr()?;
                self.expect(&TokenKind::LBrace, "open for body with `{`")?;
                let body = self.parse_block_body()?;
                self.expect(&TokenKind::RBrace, "close for body with `}`")?;
                Ok(Stmt::For { var, iter, body, span })
            }
            TokenKind::Import => {
                self.advance();
                let mut path = Vec::new();
                loop {
                    match self.peek().clone() {
                        TokenKind::Ident(n) => { self.advance(); path.push(n); }
                        _ => break,
                    }
                    if !self.eat(&TokenKind::Dot) { break; }
                }
                let alias = if let TokenKind::Ident(kw) = self.peek().clone() {
                    if kw == "as" { self.advance();
                        match self.peek().clone() {
                            TokenKind::Ident(a) => { self.advance(); Some(a) }
                            _ => None,
                        }
                    } else { None }
                } else { None };
                self.eat(&TokenKind::Semicolon);
                Ok(Stmt::Import { path, alias, span })
            }
            _ => {
                let expr = self.parse_expr()?;
                self.eat(&TokenKind::Semicolon);
                Ok(Stmt::Expr(expr))
            }
        }
    }
}

// ── Expressions (Pratt / precedence climbing) ─────────────────────────────────
impl Parser {
    pub fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_assign()
    }

    fn parse_assign(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_or()?;
        if self.eat(&TokenKind::Assign) {
            let span = left.span().clone();
            let right = self.parse_assign()?;
            return Ok(Expr::Assign {
                target: Box::new(left),
                value:  Box::new(right),
                span,
            });
        }
        Ok(left)
    }

    fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_and()?;
        while self.eat(&TokenKind::Or) {
            let span = left.span().clone();
            let right = self.parse_and()?;
            left = Expr::Binary { op: BinaryOp::Or, left: Box::new(left), right: Box::new(right), span };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_eq()?;
        while self.eat(&TokenKind::And) {
            let span = left.span().clone();
            let right = self.parse_eq()?;
            left = Expr::Binary { op: BinaryOp::And, left: Box::new(left), right: Box::new(right), span };
        }
        Ok(left)
    }

    fn parse_eq(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_cmp()?;
        loop {
            let op = match self.peek() {
                TokenKind::Eq    => BinaryOp::Eq,
                TokenKind::BangEq => BinaryOp::NotEq,
                _ => break,
            };
            self.advance();
            let span = left.span().clone();
            let right = self.parse_cmp()?;
            left = Expr::Binary { op, left: Box::new(left), right: Box::new(right), span };
        }
        Ok(left)
    }

    fn parse_cmp(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_add()?;
        loop {
            let op = match self.peek() {
                TokenKind::Lt   => BinaryOp::Lt,
                TokenKind::Gt   => BinaryOp::Gt,
                TokenKind::LtEq => BinaryOp::LtEq,
                TokenKind::GtEq => BinaryOp::GtEq,
                _ => break,
            };
            self.advance();
            let span = left.span().clone();
            let right = self.parse_add()?;
            left = Expr::Binary { op, left: Box::new(left), right: Box::new(right), span };
        }
        Ok(left)
    }

    fn parse_add(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                TokenKind::Plus  => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.advance();
            let span = left.span().clone();
            let right = self.parse_mul()?;
            left = Expr::Binary { op, left: Box::new(left), right: Box::new(right), span };
        }
        Ok(left)
    }

    fn parse_mul(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                TokenKind::Star    => BinaryOp::Mul,
                TokenKind::Slash   => BinaryOp::Div,
                TokenKind::Percent => BinaryOp::Mod,
                _ => break,
            };
            self.advance();
            let span = left.span().clone();
            let right = self.parse_unary()?;
            left = Expr::Binary { op, left: Box::new(left), right: Box::new(right), span };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        let span = self.span();
        if self.eat(&TokenKind::Bang) {
            return Ok(Expr::Unary { op: UnaryOp::Not, expr: Box::new(self.parse_unary()?), span });
        }
        if self.eat(&TokenKind::Minus) {
            return Ok(Expr::Unary { op: UnaryOp::Neg, expr: Box::new(self.parse_unary()?), span });
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;
        loop {
            let span = expr.span().clone();
            match self.peek().clone() {
                TokenKind::LParen => {
                    self.advance();
                    let args = self.parse_call_args()?;
                    self.expect(&TokenKind::RParen, "close call with `)`")?;
                    expr = Expr::Call { callee: Box::new(expr), args, span };
                }
                TokenKind::Dot => {
                    self.advance();
                    // .await special case
                    if let TokenKind::Await = self.peek() {
                        self.advance();
                        expr = Expr::Await { expr: Box::new(expr), span };
                        continue;
                    }
                    let field = match self.peek().clone() {
                        TokenKind::Ident(n) => { self.advance(); n }
                        _ => return Err(self.err("expected field name after `.`",
                            "write: object.field")),
                    };
                    expr = Expr::Field { object: Box::new(expr), field, span };
                }
                TokenKind::LBracket => {
                    self.advance();
                    let index = self.parse_expr()?;
                    self.expect(&TokenKind::RBracket, "close index with `]`")?;
                    expr = Expr::Index { object: Box::new(expr), index: Box::new(index), span };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_call_args(&mut self) -> Result<Vec<Expr>, ParseError> {
        let mut args = Vec::new();
        while !matches!(self.peek(), TokenKind::RParen | TokenKind::Eof) {
            args.push(self.parse_expr()?);
            if !self.eat(&TokenKind::Comma) { break; }
        }
        Ok(args)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let span = self.span();
        match self.peek().clone() {
            TokenKind::Int(n)   => { self.advance(); Ok(Expr::Int(n, span)) }
            TokenKind::Float(f) => { self.advance(); Ok(Expr::Float(f, span)) }
            TokenKind::Bool(b)  => { self.advance(); Ok(Expr::Bool(b, span)) }
            TokenKind::Str(s)   => { self.advance(); Ok(Expr::Str(s, span)) }

            TokenKind::Ident(name) => {
                self.advance();
                // struct literal: Name { field: val }
                if self.check(&TokenKind::LBrace) {
                    self.advance();
                    let mut fields = Vec::new();
                    while !matches!(self.peek(), TokenKind::RBrace | TokenKind::Eof) {
                        let fname = match self.peek().clone() {
                            TokenKind::Ident(f) => { self.advance(); f }
                            _ => return Err(self.err("expected field name", "")),
                        };
                        self.expect(&TokenKind::Colon, "field must be followed by `:`")?;
                        let val = self.parse_expr()?;
                        fields.push((fname, val));
                        self.eat(&TokenKind::Comma);
                    }
                    self.expect(&TokenKind::RBrace, "close struct literal with `}`")?;
                    return Ok(Expr::StructLit { name, fields, span });
                }
                Ok(Expr::Var(name, span))
            }

            TokenKind::LParen => {
                self.advance();
                if self.check(&TokenKind::RParen) { self.advance(); return Ok(Expr::Tuple(vec![], span)); }
                let first = self.parse_expr()?;
                if self.eat(&TokenKind::Comma) {
                    let mut elems = vec![first];
                    while !matches!(self.peek(), TokenKind::RParen | TokenKind::Eof) {
                        elems.push(self.parse_expr()?);
                        if !self.eat(&TokenKind::Comma) { break; }
                    }
                    self.expect(&TokenKind::RParen, "close tuple with `)`")?;
                    return Ok(Expr::Tuple(elems, span));
                }
                self.expect(&TokenKind::RParen, "close parenthesised expression with `)`")?;
                Ok(first)
            }

            TokenKind::LBracket => {
                self.advance();
                let mut elems = Vec::new();
                while !matches!(self.peek(), TokenKind::RBracket | TokenKind::Eof) {
                    elems.push(self.parse_expr()?);
                    if !self.eat(&TokenKind::Comma) { break; }
                }
                self.expect(&TokenKind::RBracket, "close array with `]`")?;
                Ok(Expr::Array(elems, span))
            }

            TokenKind::LBrace => {
                self.advance();
                let stmts = self.parse_block_body()?;
                // optional trailing expression (no semicolon)
                self.expect(&TokenKind::RBrace, "close block with `}`")?;
                Ok(Expr::Block(stmts, None, span))
            }

            TokenKind::If => {
                self.advance();
                let cond = self.parse_expr()?;
                self.expect(&TokenKind::LBrace, "open if body with `{`")?;
                let then_stmts = self.parse_block_body()?;
                self.expect(&TokenKind::RBrace, "close if body with `}`")?;
                let then_expr = Expr::Block(then_stmts, None, span.clone());
                let else_branch = if self.eat(&TokenKind::Else) {
                    self.expect(&TokenKind::LBrace, "open else body with `{`")?;
                    let else_stmts = self.parse_block_body()?;
                    self.expect(&TokenKind::RBrace, "close else body with `}`")?;
                    Some(Box::new(Expr::Block(else_stmts, None, span.clone())))
                } else { None };
                Ok(Expr::If { cond: Box::new(cond), then_branch: Box::new(then_expr), else_branch, span })
            }

            TokenKind::Match => {
                self.advance();
                let subject = self.parse_expr()?;
                self.expect(&TokenKind::LBrace, "open match with `{`")?;
                let mut arms = Vec::new();
                while !matches!(self.peek(), TokenKind::RBrace | TokenKind::Eof) {
                    let pattern = self.parse_pattern()?;
                    self.expect(&TokenKind::FatArrow, "expected `=>` after pattern")?;
                    let body = self.parse_expr()?;
                    self.eat(&TokenKind::Comma);
                    arms.push(MatchArm { pattern, body });
                }
                self.expect(&TokenKind::RBrace, "close match with `}`")?;
                Ok(Expr::Match { subject: Box::new(subject), arms, span })
            }

            _ => Err(ParseError {
                msg:  format!("unexpected token `{:?}` in expression", self.peek()),
                span,
                hint: Some("check the expression syntax".to_string()),
            }),
        }
    }

    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        let span = self.span();
        match self.peek().clone() {
            TokenKind::Ident(n) if n == "_" => { self.advance(); Ok(Pattern::Wildcard(span)) }
            TokenKind::Ident(n) => { self.advance(); Ok(Pattern::Binding(n, span)) }
            TokenKind::Int(_) | TokenKind::Float(_) | TokenKind::Bool(_) | TokenKind::Str(_) => {
                let e = self.parse_primary()?;
                Ok(Pattern::Literal(e))
            }
            _ => Err(self.err("unsupported pattern", "try a literal, identifier, or `_`")),
        }
    }

    fn parse_type_expr(&mut self) -> Result<TypeExpr, ParseError> {
        match self.peek().clone() {
            TokenKind::Ident(name) => {
                self.advance();
                // generic e.g. List<Int>
                if self.eat(&TokenKind::Lt) {
                    let mut args = Vec::new();
                    loop {
                        args.push(self.parse_type_expr()?);
                        if !self.eat(&TokenKind::Comma) { break; }
                    }
                    self.expect(&TokenKind::Gt, "close generic with `>`")?;
                    return Ok(TypeExpr::Generic(name, args));
                }
                Ok(TypeExpr::Named(name))
            }
            TokenKind::LParen => {
                self.advance();
                let mut types = Vec::new();
                while !matches!(self.peek(), TokenKind::RParen | TokenKind::Eof) {
                    types.push(self.parse_type_expr()?);
                    if !self.eat(&TokenKind::Comma) { break; }
                }
                self.expect(&TokenKind::RParen, "close tuple type with `)`")?;
                Ok(TypeExpr::Tuple(types))
            }
            _ => Err(self.err("expected type expression",
                "e.g. `Int`, `String`, `List<Int>`")),
        }
    }
}
