// ─────────────────────────────────────────────────────────────────────────────
// LEXER  (source text → token stream)
// ─────────────────────────────────────────────────────────────────────────────
//
// Design decisions recorded here so they don't become silent assumptions:
//
//  [D1] Source storage: &str + byte-index pos, NOT Vec<char>.
//       Vec<char> costs 4 bytes per character even for pure ASCII.
//       We decode chars on demand via str::chars() from the current position.
//
//  [D2] Filename storage: Arc<str>, NOT String.
//       Every token span would otherwise heap-allocate a copy of the filename.
//       Arc<str> makes Span::clone() a ref-count bump instead of a memcpy.
//
//  [D3] Identifier character set: ASCII only (is_ascii_alphabetic / _).
//       Unicode identifiers are powerful but impose ordering/normalisation
//       questions that are out of scope right now.  The decision is explicit
//       so it can be changed in one place (read_ident_or_keyword) later.
//
//  [D4] Compound-assignment operators (+=, -=, *=, /=, %=) and null-coalescing
//       (??) are part of the token set.  They are lexed, even if the parser
//       does not yet use all of them.
//
//  [D5] Error recovery: tokenize() returns (Vec<Token>, Vec<LexError>).
//       Lexing continues after most errors so the user sees all problems in
//       one pass.  Recovery strategy per error kind is documented inline.
//
//  [D6] Block-comment nesting is real.  A depth counter is maintained so
//       /* /* inner */ still open */ is only closed at depth 0.
//       The diagnostic no longer lies about this behaviour.
//
// ─────────────────────────────────────────────────────────────────────────────

use std::sync::Arc;

// ── ANSI colour helpers ───────────────────────────────────────────────────────

struct Colour;
impl Colour {
    fn enabled() -> bool { std::env::var("NO_COLOR").is_err() }

    fn red(s: &str)    -> String { Self::wrap(s, "31") }
    fn yellow(s: &str) -> String { Self::wrap(s, "33") }
    fn cyan(s: &str)   -> String { Self::wrap(s, "36") }
    fn bold(s: &str)   -> String { Self::wrap(s, "1")  }
    fn dim(s: &str)    -> String { Self::wrap(s, "2")  }
    fn blue(s: &str)   -> String { Self::wrap(s, "34") }
    fn green(s: &str)  -> String { Self::wrap(s, "32") }

    fn wrap(s: &str, code: &str) -> String {
        if Self::enabled() { format!("\x1b[{}m{}\x1b[0m", code, s) }
        else                { s.to_string() }
    }
}

// ── Span ─────────────────────────────────────────────────────────────────────
//
// [D2] `file` is an Arc<str>.  All spans produced by one Lexer share the same
// Arc, so Clone is cheap (one atomic increment, no allocation).

#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub file:      Arc<str>,
    pub line:      usize,
    pub col_start: usize,   // 1-based, character columns (not bytes)
    pub col_end:   usize,   // 1-based exclusive
}

impl Span {
    fn point(file: &Arc<str>, line: usize, col: usize) -> Self {
        Self { file: Arc::clone(file), line, col_start: col, col_end: col + 1 }
    }
    fn range(file: &Arc<str>, line: usize, col_start: usize, col_end: usize) -> Self {
        Self { file: Arc::clone(file), line, col_start, col_end }
    }
}

// ── Tokens ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    Int(i64), Float(f64), Bool(bool), Str(String),

    // Identifiers & keywords
    Ident(String),
    Let, Fn, Return, If, Else, While, For, In,
    Import, Pub, Struct, Enum, Match, Async, Await,

    // Arithmetic
    Plus, Minus, Star, Slash, Percent,
    // [D4] Compound assignment
    PlusEq, MinusEq, StarEq, SlashEq, PercentEq,

    // Comparison / logical
    Eq, BangEq, Lt, Gt, LtEq, GtEq,
    And, Or, Bang,

    // Assignment / arrows
    Assign,   // =
    Arrow,    // ->
    FatArrow, // =>

    // Access / ranges
    Dot, DotDot,

    // [D4] Null-coalescing / optional chaining
    Question,         // ?
    QuestionQuestion, // ??

    // Delimiters
    LParen, RParen,
    LBrace, RBrace,
    LBracket, RBracket,
    Comma, Colon, Semicolon,

    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

// ── LexError ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum NumericError {
    IntegerOverflow,
    MultipleDecimalPoints,
    TrailingDot,
    MalformedLiteral,
}

#[derive(Debug, Clone)]
pub enum BitwiseContext { AfterAmpersand, AfterPipe }

#[derive(Debug, Clone)]
pub enum LexError {
    UnexpectedChar {
        ch:   char,
        span: Span,
        prev: Option<char>,
    },
    UnterminatedString {
        span:   Span,
        opened: Span,
    },
    UnterminatedBlockComment {
        span:      Span,
        opened:    Span,
        /// Maximum nesting depth seen before EOF, for diagnostic context.
        max_depth: usize,
    },
    InvalidEscape {
        ch:          char,
        span:        Span,
        partial_str: String,
    },
    MalformedNumber {
        raw:    String,
        span:   Span,
        reason: NumericError,
    },
    SingleBitwiseOp {
        span:    Span,
        context: BitwiseContext,
    },
    UnexpectedEof {
        span: Span,
    },
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LexError::UnexpectedChar { ch, span, .. } =>
                write!(f, "error[L001]: unexpected character `{}` at {}:{}:{}",
                    ch, span.file, span.line, span.col_start),
            LexError::UnterminatedString { span, .. } =>
                write!(f, "error[L002]: unterminated string at {}:{}:{}",
                    span.file, span.line, span.col_start),
            LexError::UnterminatedBlockComment { span, .. } =>
                write!(f, "error[L003]: unterminated block comment at {}:{}:{}",
                    span.file, span.line, span.col_start),
            LexError::InvalidEscape { ch, span, .. } =>
                write!(f, "error[L004]: invalid escape `\\{}` at {}:{}:{}",
                    ch, span.file, span.line, span.col_start),
            LexError::MalformedNumber { raw, span, .. } =>
                write!(f, "error[L005]: malformed number `{}` at {}:{}:{}",
                    raw, span.file, span.line, span.col_start),
            LexError::SingleBitwiseOp { span, .. } =>
                write!(f, "error[L006]: lone bitwise operator at {}:{}:{}",
                    span.file, span.line, span.col_start),
            LexError::UnexpectedEof { span } =>
                write!(f, "error[L007]: unexpected EOF at {}:{}:{}",
                    span.file, span.line, span.col_start),
        }
    }
}

impl std::error::Error for LexError {}

// ── Lexer ─────────────────────────────────────────────────────────────────────
//
// [D1] `source` is &str and `pos` is a *byte* index into it.
// Characters are decoded only when peek()/advance() are called.
// For ASCII source (the common case) each character is exactly 1 byte, so
// this is also cache-friendly: no pointer-chasing through a Vec heap object.

pub struct Lexer<'src> {
    source:    &'src str,
    pos:       usize,       // byte index, always on a char boundary
    line:      usize,
    col:       usize,       // 1-based character column (not byte offset)
    file:      Arc<str>,
    prev_char: Option<char>,
}

impl<'src> Lexer<'src> {
    pub fn new(source: &'src str, file: &str) -> Self {
        Self {
            source,
            pos: 0,
            line: 1,
            col: 1,
            file: Arc::from(file),
            prev_char: None,
        }
    }

    // ── low-level cursor ──────────────────────────────────────────────────────

    /// Return the next character without consuming it.
    fn peek(&self) -> Option<char> {
        self.source[self.pos..].chars().next()
    }

    /// Return the character *after* the next one without consuming either.
    fn peek2(&self) -> Option<char> {
        let mut it = self.source[self.pos..].chars();
        it.next()?;
        it.next()
    }

    /// Consume and return the next character, advancing `pos` by its UTF-8
    /// byte length (1 for ASCII, 2-4 for multibyte).
    fn advance(&mut self) -> Option<char> {
        let ch = self.source[self.pos..].chars().next()?;
        self.pos += ch.len_utf8();          // [D1] byte-accurate advance
        if ch == '\n' { self.line += 1; self.col = 1; }
        else          { self.col  += 1; }
        self.prev_char = Some(ch);
        Some(ch)
    }

    fn span_here(&self) -> Span {
        Span::point(&self.file, self.line, self.col)
    }

    fn span_from(&self, col_start: usize, line: usize) -> Span {
        Span::range(&self.file, line, col_start, self.col)
    }

    // ── whitespace / comments ─────────────────────────────────────────────────

    fn skip_whitespace_and_comments(&mut self) -> Option<LexError> {
        loop {
            match self.peek() {
                Some(c) if c.is_ascii_whitespace() => { self.advance(); }

                // Line comment: consume through end-of-line.
                Some('/') if self.peek2() == Some('/') => {
                    while self.peek().map(|c| c != '\n').unwrap_or(false) {
                        self.advance();
                    }
                }

                // Block comment: [D6] real nesting via depth counter.
                Some('/') if self.peek2() == Some('*') => {
                    let opened = self.span_here();
                    self.advance(); self.advance(); // consume '/' '*'
                    let mut depth     = 1usize;
                    let mut max_depth = 1usize;
                    loop {
                        match (self.peek(), self.peek2()) {
                            (Some('/'), Some('*')) => {
                                self.advance(); self.advance();
                                depth    += 1;
                                max_depth = max_depth.max(depth);
                            }
                            (Some('*'), Some('/')) => {
                                self.advance(); self.advance();
                                depth -= 1;
                                if depth == 0 { break; }
                            }
                            (None, _) => {
                                return Some(LexError::UnterminatedBlockComment {
                                    span: self.span_here(),
                                    opened,
                                    max_depth,
                                });
                            }
                            _ => { self.advance(); }
                        }
                    }
                }

                _ => break,
            }
        }
        None
    }

    // ── string literals ───────────────────────────────────────────────────────
    //
    // [D5] Recovery inside strings: an invalid escape is recorded but lexing
    // continues.  The raw character is substituted so the token remains useful.
    //
    // Returns Ok((value, inner_errors)) on a terminated string (possibly with
    // recovered escape errors), or Err(LexError) for unterminated.

    fn read_string(
        &mut self,
        open_span: Span,
    ) -> Result<(String, Vec<LexError>), LexError> {
        let mut s    = String::new();
        let mut errs = Vec::new();

        loop {
            match self.peek() {
                None | Some('\n') => {
                    return Err(LexError::UnterminatedString {
                        span:   self.span_here(),
                        opened: open_span,
                    });
                }
                _ => {}
            }

            match self.advance().unwrap() {
                '"'  => break,
                '\\' => {
                    let esc_span = self.span_here();
                    match self.advance() {
                        Some('n')  => s.push('\n'),
                        Some('t')  => s.push('\t'),
                        Some('"')  => s.push('"'),
                        Some('\\') => s.push('\\'),
                        Some(c) => {
                            // [D5] Record the error, substitute the raw char,
                            // and continue so subsequent errors are visible.
                            errs.push(LexError::InvalidEscape {
                                ch:          c,
                                span:        esc_span,
                                partial_str: s.clone(),
                            });
                            s.push(c);
                        }
                        None => {
                            return Err(LexError::UnterminatedString {
                                span:   self.span_here(),
                                opened: open_span,
                            });
                        }
                    }
                }
                c => s.push(c),
            }
        }
        Ok((s, errs))
    }

    // ── numeric literals ──────────────────────────────────────────────────────

    fn read_number(
        &mut self,
        first:     char,
        start_col: usize,
    ) -> Result<TokenKind, LexError> {
        let start_line = self.line;
        let mut raw       = String::from(first);
        let mut is_float  = false;
        let mut dot_count = 0u32;

        loop {
            match self.peek() {
                Some(c) if c.is_ascii_digit() => { raw.push(c); self.advance(); }
                Some('.') if self.peek2().map(|x| x.is_ascii_digit()).unwrap_or(false) => {
                    raw.push('.'); self.advance();
                    is_float  = true;
                    dot_count += 1;
                    if let Some(d) = self.peek() {
                        if d.is_ascii_digit() { raw.push(d); self.advance(); }
                    }
                }
                _ => break,
            }
        }

        let span = self.span_from(start_col, start_line);

        if dot_count > 1 {
            return Err(LexError::MalformedNumber {
                raw, span, reason: NumericError::MultipleDecimalPoints,
            });
        }

        if is_float {
            raw.parse::<f64>()
               .map(TokenKind::Float)
               .map_err(|_| LexError::MalformedNumber {
                   raw, span, reason: NumericError::MalformedLiteral,
               })
        } else {
            raw.parse::<i64>().map(TokenKind::Int).map_err(|e| {
                let reason = match e.kind() {
                    std::num::IntErrorKind::PosOverflow |
                    std::num::IntErrorKind::NegOverflow => NumericError::IntegerOverflow,
                    _                                   => NumericError::MalformedLiteral,
                };
                LexError::MalformedNumber { raw, span, reason }
            })
        }
    }

    // ── identifiers / keywords ────────────────────────────────────────────────
    //
    // [D3] ASCII-only identifiers: [A-Za-z0-9_]+ with [A-Za-z_] as first char.
    // To allow Unicode, replace is_ascii_alphanumeric() with is_alphanumeric()
    // and update the dispatch predicate in tokenize() to match.

    fn read_ident_or_keyword(&mut self, first: char) -> TokenKind {
        let mut s = String::from(first);
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' { // [D3]
                s.push(c); self.advance();
            } else {
                break;
            }
        }
        match s.as_str() {
            "let"    => TokenKind::Let,
            "fn"     => TokenKind::Fn,
            "return" => TokenKind::Return,
            "if"     => TokenKind::If,
            "else"   => TokenKind::Else,
            "while"  => TokenKind::While,
            "for"    => TokenKind::For,
            "in"     => TokenKind::In,
            "true"   => TokenKind::Bool(true),
            "false"  => TokenKind::Bool(false),
            "import" => TokenKind::Import,
            "pub"    => TokenKind::Pub,
            "struct" => TokenKind::Struct,
            "enum"   => TokenKind::Enum,
            "match"  => TokenKind::Match,
            "async"  => TokenKind::Async,
            "await"  => TokenKind::Await,
            _        => TokenKind::Ident(s),
        }
    }

    // ── main tokenise loop ────────────────────────────────────────────────────
    //
    // [D5] Returns ALL tokens AND ALL errors found in one pass.
    // Recovery strategy per error kind:
    //
    //   UnexpectedChar           -> skip 1 char, continue
    //   InvalidEscape (in str)   -> substitute raw char, finish string, continue
    //   UnterminatedString       -> no token; pos is at EOL/EOF, continue
    //   UnterminatedBlockComment -> pos is at EOF; push error, break
    //   MalformedNumber          -> no token; continue
    //   SingleBitwiseOp          -> no token (char consumed); continue
    //   UnexpectedEof            -> break (at EOF)

    pub fn tokenize(&mut self) -> (Vec<Token>, Vec<LexError>) {
        let mut tokens = Vec::new();
        let mut errors = Vec::new();

        loop {
            if let Some(err) = self.skip_whitespace_and_comments() {
                errors.push(err);
                tokens.push(Token { kind: TokenKind::Eof, span: self.span_here() });
                break;
            }

            let start_col  = self.col;
            let start_line = self.line;
            let span       = self.span_here();

            let ch = match self.advance() {
                None => {
                    tokens.push(Token { kind: TokenKind::Eof, span });
                    break;
                }
                Some(c) => c,
            };

            let kind: TokenKind = match ch {

                // ── arithmetic (with [D4] compound-assignment) ────────────────
                '+' if self.peek() == Some('=') => { self.advance(); TokenKind::PlusEq }
                '+' => TokenKind::Plus,

                '-' if self.peek() == Some('>') => { self.advance(); TokenKind::Arrow }
                '-' if self.peek() == Some('=') => { self.advance(); TokenKind::MinusEq }
                '-' => TokenKind::Minus,

                '*' if self.peek() == Some('=') => { self.advance(); TokenKind::StarEq }
                '*' => TokenKind::Star,

                '/' if self.peek() == Some('=') => { self.advance(); TokenKind::SlashEq }
                '/' => TokenKind::Slash,

                '%' if self.peek() == Some('=') => { self.advance(); TokenKind::PercentEq }
                '%' => TokenKind::Percent,

                // ── comparison / logical ──────────────────────────────────────
                '=' if self.peek() == Some('=') => { self.advance(); TokenKind::Eq }
                '=' if self.peek() == Some('>') => { self.advance(); TokenKind::FatArrow }
                '=' => TokenKind::Assign,

                '!' if self.peek() == Some('=') => { self.advance(); TokenKind::BangEq }
                '!' => TokenKind::Bang,

                '<' if self.peek() == Some('=') => { self.advance(); TokenKind::LtEq }
                '<' => TokenKind::Lt,

                '>' if self.peek() == Some('=') => { self.advance(); TokenKind::GtEq }
                '>' => TokenKind::Gt,

                // [D5] Lone `&` / `|` -> error + skip, continue.
                '&' if self.peek() == Some('&') => { self.advance(); TokenKind::And }
                '&' => {
                    errors.push(LexError::SingleBitwiseOp {
                        span: Span::range(&self.file, start_line, start_col, start_col + 1),
                        context: BitwiseContext::AfterAmpersand,
                    });
                    continue;
                }

                '|' if self.peek() == Some('|') => { self.advance(); TokenKind::Or }
                '|' => {
                    errors.push(LexError::SingleBitwiseOp {
                        span: Span::range(&self.file, start_line, start_col, start_col + 1),
                        context: BitwiseContext::AfterPipe,
                    });
                    continue;
                }

                // [D4] `?` and `??`
                '?' if self.peek() == Some('?') => { self.advance(); TokenKind::QuestionQuestion }
                '?' => TokenKind::Question,

                '.' if self.peek() == Some('.') => { self.advance(); TokenKind::DotDot }
                '.' => TokenKind::Dot,

                // ── delimiters ────────────────────────────────────────────────
                '(' => TokenKind::LParen,
                ')' => TokenKind::RParen,
                '{' => TokenKind::LBrace,
                '}' => TokenKind::RBrace,
                '[' => TokenKind::LBracket,
                ']' => TokenKind::RBracket,
                ',' => TokenKind::Comma,
                ':' => TokenKind::Colon,
                ';' => TokenKind::Semicolon,

                // ── string literals ───────────────────────────────────────────
                '"' => {
                    let open_span = Span::point(&self.file, start_line, start_col);
                    match self.read_string(open_span) {
                        Ok((s, inner_errs)) => {
                            errors.extend(inner_errs);
                            TokenKind::Str(s)
                        }
                        Err(e) => {
                            // [D5] Unterminated: no token; pos is at EOL, safe to continue.
                            errors.push(e);
                            continue;
                        }
                    }
                }

                // ── numeric literals ──────────────────────────────────────────
                c if c.is_ascii_digit() => {
                    match self.read_number(c, start_col) {
                        Ok(k)  => k,
                        Err(e) => { errors.push(e); continue; }
                    }
                }

                // ── identifiers / keywords ─────────────────────────────────
                // [D3] ASCII-only: is_ascii_alphabetic() + '_'
                c if c.is_ascii_alphabetic() || c == '_' =>
                    self.read_ident_or_keyword(c),

                // ── unrecognised character ────────────────────────────────────
                other => {
                    let prev = self.prev_char;
                    errors.push(LexError::UnexpectedChar {
                        ch:   other,
                        span: Span::range(&self.file, start_line, start_col,
                                          start_col + other.len_utf8()),
                        prev,
                    });
                    continue; // [D5] skip 1 char and continue
                }
            };

            let full_span = Span::range(&self.file, start_line, start_col, self.col);
            tokens.push(Token { kind, span: full_span });
        }

        (tokens, errors)
    }
}

// ── Diagnostic renderer ───────────────────────────────────────────────────────

pub struct Diagnostic<'src> {
    pub error:        LexError,
    pub source_lines: &'src [&'src str],
}

impl<'src> Diagnostic<'src> {
    pub fn new(error: LexError, source_lines: &'src [&'src str]) -> Self {
        Self { error, source_lines }
    }
}

impl<'src> std::fmt::Display for Diagnostic<'src> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        render_diagnostic(f, &self.error, self.source_lines)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rendering internals
// ─────────────────────────────────────────────────────────────────────────────

fn render_diagnostic(f: &mut std::fmt::Formatter<'_>, error: &LexError, src: &[&str])
    -> std::fmt::Result
{
    match error {
        LexError::UnexpectedChar { ch, span, prev } =>
            render_unexpected_char(f, *ch, span, *prev, src),
        LexError::UnterminatedString { span, opened } =>
            render_unterminated_string(f, span, opened, src),
        LexError::UnterminatedBlockComment { span, opened, max_depth } =>
            render_unterminated_block_comment(f, span, opened, *max_depth, src),
        LexError::InvalidEscape { ch, span, partial_str } =>
            render_invalid_escape(f, *ch, span, partial_str, src),
        LexError::MalformedNumber { raw, span, reason } =>
            render_malformed_number(f, raw, span, reason, src),
        LexError::SingleBitwiseOp { span, context } =>
            render_single_bitwise_op(f, span, context, src),
        LexError::UnexpectedEof { span } =>
            render_unexpected_eof(f, span, src),
    }
}

fn header(f: &mut std::fmt::Formatter<'_>, code: &str, msg: &str) -> std::fmt::Result {
    writeln!(f, "{}: {}",
        Colour::bold(&Colour::red(&format!("error[{}]", code))),
        Colour::bold(msg))
}

fn location_arrow(f: &mut std::fmt::Formatter<'_>, span: &Span) -> std::fmt::Result {
    writeln!(f, "  {} {}:{}:{}",
        Colour::bold(&Colour::blue("-->")),
        span.file, span.line, span.col_start)
}

fn gutter_blank(f: &mut std::fmt::Formatter<'_>, w: usize) -> std::fmt::Result {
    writeln!(f, "{:>width$} {}", Colour::bold(&Colour::blue("|")), "", width = w + 1)
}

fn gutter_source(f: &mut std::fmt::Formatter<'_>, n: usize, line: &str, w: usize)
    -> std::fmt::Result
{
    writeln!(f, "{} {} {}",
        Colour::bold(&Colour::blue(&format!("{:>w$}", n, w = w))),
        Colour::bold(&Colour::blue("|")),
        line)
}

fn gutter_caret(
    f: &mut std::fmt::Formatter<'_>, w: usize,
    col: usize, len: usize, msg: &str, colour: fn(&str) -> String,
) -> std::fmt::Result {
    let indent = " ".repeat(col.saturating_sub(1));
    let carets = colour(&"^".repeat(len.max(1)));
    let tail   = if msg.is_empty() { String::new() } else { format!("  {}", colour(msg)) };
    writeln!(f, "{:>width$} {}{}{}", Colour::bold(&Colour::blue("|")), indent, carets, tail, width = w + 1)
}

fn gutter_secondary(
    f: &mut std::fmt::Formatter<'_>, w: usize,
    col: usize, len: usize, msg: &str,
) -> std::fmt::Result {
    let indent = " ".repeat(col.saturating_sub(1));
    let dashes = Colour::dim(&Colour::blue(&"-".repeat(len.max(1))));
    let tail   = if msg.is_empty() { String::new() }
                 else { format!("  {}", Colour::dim(&Colour::blue(msg))) };
    writeln!(f, "{:>width$} {}{}{}", Colour::bold(&Colour::blue("|")), indent, dashes, tail, width = w + 1)
}

fn annotation(f: &mut std::fmt::Formatter<'_>, w: usize, label: &str, text: &str)
    -> std::fmt::Result
{
    let lbl = match label {
        "why"  => Colour::bold(&Colour::yellow("why: ")),
        "note" => Colour::bold(&Colour::cyan("note:")),
        "help" => Colour::bold(&Colour::green("help:")),
        _      => Colour::bold(label),
    };
    writeln!(f, "{:>width$} {} {} {}", Colour::dim(" "), Colour::bold(&Colour::blue("=")), lbl, text, width = w + 1)
}

fn get_line<'a>(src: &'a [&'a str], lineno: usize) -> Option<&'a str> {
    src.get(lineno.saturating_sub(1)).copied()
}

fn digit_width(n: usize) -> usize {
    if n == 0 { 1 } else { (n as f64).log10().floor() as usize + 1 }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max { s.to_string() }
    else { format!("{}...", s.chars().take(max).collect::<String>()) }
}

// ── per-variant renderers ─────────────────────────────────────────────────────

fn render_unexpected_char(
    f: &mut std::fmt::Formatter<'_>,
    ch: char, span: &Span, prev: Option<char>, src: &[&str],
) -> std::fmt::Result {
    let w = digit_width(span.line);
    header(f, "L001", &format!("unexpected character `{}`", ch))?;
    location_arrow(f, span)?;
    gutter_blank(f, w)?;
    if let Some(line) = get_line(src, span.line) {
        gutter_source(f, span.line, line, w)?;
        gutter_caret(f, w, span.col_start, 1, "not a recognised character", Colour::red)?;
    }
    gutter_blank(f, w)?;

    annotation(f, w, "why",
        &format!("`{}` ({}) is not part of the language's character set.", ch, unicode_name(ch)))?;

    match prev {
        Some('&') => annotation(f, w, "note",
            "a lone `&` appeared before this; did you mean `&&` (logical AND)?")?,
        Some('|') => annotation(f, w, "note",
            "a lone `|` appeared before this; did you mean `||` (logical OR)?")?,
        _ => {}
    }

    match ch {
        '@'  => annotation(f, w, "help", "for attributes this language will use `#[...]` syntax.")?,
        '#'  => annotation(f, w, "help", "for comments use `//` or `/* ... */`.")?,
        '$'  => annotation(f, w, "help", "identifiers must start with `[A-Za-z_]`.")?,
        '`'  => annotation(f, w, "help", "use double-quoted strings `\"...\"` instead of backticks.")?,
        '\'' => annotation(f, w, "help", "there are no char literals; use a one-character string `\"a\"`.")?,
        '^'  => annotation(f, w, "help", "XOR is not supported; for logical negation use `!`.")?,
        '~'  => annotation(f, w, "help", "bitwise NOT is not supported; use `!expr` for logical negation.")?,
        c if c.is_ascii_control() =>
            annotation(f, w, "help",
                &format!("non-printable control char (0x{:02X}); delete and retype surrounding text.", c as u8))?,
        c if !c.is_ascii() =>
            annotation(f, w, "note",
                &format!("Unicode U+{:04X} is only valid inside string literals.", c as u32))?,
        _ => annotation(f, w, "note",
            "valid characters: letters A-Za-z, digits 0-9, `_`, `\"`, and operator/delimiter symbols")?,
    }
    Ok(())
}

fn render_unterminated_string(
    f: &mut std::fmt::Formatter<'_>,
    span: &Span, opened: &Span, src: &[&str],
) -> std::fmt::Result {
    let w = digit_width(span.line.max(opened.line));
    header(f, "L002", "unterminated string literal")?;
    location_arrow(f, span)?;
    gutter_blank(f, w)?;

    if opened.line != span.line {
        if let Some(line) = get_line(src, opened.line) {
            gutter_source(f, opened.line, line, w)?;
            gutter_secondary(f, w, opened.col_start, 1, "string opened here")?;
            gutter_blank(f, w)?;
        }
    }
    if let Some(line) = get_line(src, span.line) {
        gutter_source(f, span.line, line, w)?;
        gutter_caret(f, w, span.col_start, 1, "string is still open here", Colour::red)?;
    }
    gutter_blank(f, w)?;

    annotation(f, w, "why",  "the opening `\"` has no matching closing `\"`.")?;
    annotation(f, w, "note", "string literals cannot span multiple lines.")?;
    annotation(f, w, "help", "add a closing `\"`, or use `\\n` to embed a newline inside the string.")?;
    Ok(())
}

fn render_unterminated_block_comment(
    f: &mut std::fmt::Formatter<'_>,
    span: &Span, opened: &Span, max_depth: usize, src: &[&str],
) -> std::fmt::Result {
    let w = digit_width(span.line.max(opened.line));
    header(f, "L003", "unterminated block comment")?;
    location_arrow(f, span)?;
    gutter_blank(f, w)?;

    if let Some(line) = get_line(src, opened.line) {
        gutter_source(f, opened.line, line, w)?;
        gutter_secondary(f, w, opened.col_start, 2, "block comment opened here with `/*`")?;
        gutter_blank(f, w)?;
    }
    if span.line != opened.line {
        if let Some(line) = get_line(src, span.line) {
            gutter_source(f, span.line, line, w)?;
        }
    }
    gutter_caret(f, w, span.col_start, 1, "EOF reached here", Colour::red)?;
    gutter_blank(f, w)?;

    annotation(f, w, "why",
        "a `/*` block comment was opened but the file ended before a matching `*/`.")?;

    // [D6] Honest nesting message — the depth counter makes this factual.
    if max_depth > 1 {
        annotation(f, w, "note",
            &format!("this comment reached nesting depth {}; every `/*` needs its own `*/`.",
                max_depth))?;
    } else {
        annotation(f, w, "note",
            "block comments support nesting: /* /* inner */ still open */. \
             Every `/*` needs its own `*/`.")?;
    }
    annotation(f, w, "help", "add the missing `*/` to close the comment.")?;
    Ok(())
}

fn render_invalid_escape(
    f: &mut std::fmt::Formatter<'_>,
    ch: char, span: &Span, partial: &str, src: &[&str],
) -> std::fmt::Result {
    let w = digit_width(span.line);
    header(f, "L004", &format!("invalid escape sequence `\\{}`", ch))?;
    location_arrow(f, span)?;
    gutter_blank(f, w)?;
    if let Some(line) = get_line(src, span.line) {
        gutter_source(f, span.line, line, w)?;
        gutter_caret(f, w, span.col_start, 2, "unrecognised escape", Colour::red)?;
    }
    gutter_blank(f, w)?;

    annotation(f, w, "why",  &format!("`\\{}` is not a defined escape sequence.", ch))?;
    annotation(f, w, "note", "valid sequences: `\\n` `\\t` `\\\"` `\\\\`")?;
    let hint = match ch {
        'r'      => "carriage-return (`\\r`) is not supported; use `\\n` for a line-break.",
        '0'      => "`\\0` (null byte) is not supported.",
        'u' | 'x'=> "unicode/hex escapes are not yet supported.",
        '\''     => "single-quote needs no escaping inside double-quoted strings.",
        _        => "to include a literal backslash, write `\\\\`.",
    };
    annotation(f, w, "help", hint)?;
    if !partial.is_empty() {
        annotation(f, w, "note",
            &format!("string content before this escape: \"{}\"", truncate(partial, 40)))?;
    }
    Ok(())
}

fn render_malformed_number(
    f: &mut std::fmt::Formatter<'_>,
    raw: &str, span: &Span, reason: &NumericError, src: &[&str],
) -> std::fmt::Result {
    let w = digit_width(span.line);
    header(f, "L005", &format!("malformed numeric literal `{}`", raw))?;
    location_arrow(f, span)?;
    gutter_blank(f, w)?;
    if let Some(line) = get_line(src, span.line) {
        gutter_source(f, span.line, line, w)?;
        gutter_caret(f, w, span.col_start, raw.len(), "here", Colour::red)?;
    }
    gutter_blank(f, w)?;

    match reason {
        NumericError::IntegerOverflow => {
            annotation(f, w, "why",
                &format!("`{}` exceeds the i64 maximum (9_223_372_036_854_775_807).", raw))?;
            annotation(f, w, "help", "add `.0` to store it as a float, or reduce the value.")?;
        }
        NumericError::MultipleDecimalPoints => {
            annotation(f, w, "why",  "a float literal may have at most one decimal point.")?;
            annotation(f, w, "note", "this looks like a version number or IP address.")?;
            annotation(f, w, "help", "for a range expression write `start..end`.")?;
        }
        NumericError::TrailingDot => {
            annotation(f, w, "why",  "a float literal needs at least one digit after the `.`.")?;
            annotation(f, w, "help",
                &format!("write `{}.0` instead.", raw.trim_end_matches('.')))?;
        }
        NumericError::MalformedLiteral => {
            annotation(f, w, "why",  "the literal could not be parsed.")?;
            annotation(f, w, "help", "integers: plain decimal digits. Floats: digits `.` digits.")?;
        }
    }
    Ok(())
}

fn render_single_bitwise_op(
    f: &mut std::fmt::Formatter<'_>,
    span: &Span, context: &BitwiseContext, src: &[&str],
) -> std::fmt::Result {
    let (ch, op, full) = match context {
        BitwiseContext::AfterAmpersand => ('&', "&&", "logical AND"),
        BitwiseContext::AfterPipe      => ('|', "||", "logical OR"),
    };
    let w = digit_width(span.line);
    header(f, "L006", &format!("lone `{}` is not a valid operator", ch))?;
    location_arrow(f, span)?;
    gutter_blank(f, w)?;
    if let Some(line) = get_line(src, span.line) {
        gutter_source(f, span.line, line, w)?;
        gutter_caret(f, w, span.col_start, 1, &format!("did you mean `{}`?", op), Colour::red)?;
    }
    gutter_blank(f, w)?;
    annotation(f, w, "why",  "the language has no bitwise operators.")?;
    annotation(f, w, "note", &format!("`{}` is the {} operator.", op, full))?;
    annotation(f, w, "help", &format!("replace `{}` with `{}`.", ch, op))?;
    Ok(())
}

fn render_unexpected_eof(
    f: &mut std::fmt::Formatter<'_>,
    span: &Span, src: &[&str],
) -> std::fmt::Result {
    let w = digit_width(span.line);
    header(f, "L007", "unexpected end of file")?;
    location_arrow(f, span)?;
    gutter_blank(f, w)?;
    if let Some(line) = get_line(src, span.line) {
        gutter_source(f, span.line, line, w)?;
        gutter_caret(f, w, line.len() + 1, 1, "file ends here", Colour::red)?;
    }
    gutter_blank(f, w)?;
    annotation(f, w, "why",  "the file ended while more tokens were expected.")?;
    annotation(f, w, "note", "a common cause is a missing `}` or an incomplete expression.")?;
    annotation(f, w, "help", "check that all `{` are closed and all statements end with `;`.")?;
    Ok(())
}

fn unicode_name(ch: char) -> &'static str {
    match ch {
        '@' => "at-sign",  '#'  => "hash",          '$'  => "dollar",
        '`' => "backtick", '\'' => "single-quote",   '^'  => "caret",
        '~' => "tilde",    '\\' => "backslash",      '\t' => "horizontal tab",
        '\r'=> "carriage return",                    '\n' => "newline",
        _   => "symbol",
    }
}

// ── Convenience wrapper ───────────────────────────────────────────────────────

pub fn lex_and_report(source: &str, file: &str) -> (Vec<Token>, Vec<LexError>) {
    let mut lexer = Lexer::new(source, file);
    let (tokens, errors) = lexer.tokenize();
    if !errors.is_empty() {
        let lines: Vec<&str> = source.lines().collect();
        for err in &errors {
            eprintln!("{}", Diagnostic::new(err.clone(), &lines));
        }
    }
    (tokens, errors)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn lex(src: &str) -> (Vec<TokenKind>, Vec<LexError>) {
        let mut l = Lexer::new(src, "test.lang");
        let (toks, errs) = l.tokenize();
        (toks.into_iter().map(|t| t.kind).collect(), errs)
    }

    fn lex_ok(src: &str) -> Vec<TokenKind> {
        let (toks, errs) = lex(src);
        assert!(errs.is_empty(), "unexpected errors: {:?}", errs);
        toks
    }

    fn lex_errs(src: &str) -> Vec<LexError> { lex(src).1 }

    // ── [D1] &str / byte-index correctness ───────────────────────────────────

    #[test]
    fn ascii_source_tokenises() {
        let toks = lex_ok("let x = 42;");
        assert_eq!(toks[0], TokenKind::Let);
        assert_eq!(toks[2], TokenKind::Assign);
        assert_eq!(toks[3], TokenKind::Int(42));
    }

    #[test]
    fn multibyte_char_inside_string() {
        // "cafe\u{301}" contains a 2-byte char; the lexer must advance by char not byte.
        let toks = lex_ok("\"caf\u{e9}\"");
        assert_eq!(toks[0], TokenKind::Str("caf\u{e9}".to_string()));
    }

    #[test]
    fn span_col_is_character_not_byte() {
        // "é" is 1 char but 2 bytes. The integer after the string should be
        // at character column 5, not byte column 6.
        let mut l = Lexer::new("\"é\" 1", "t");
        let (toks, _) = l.tokenize();
        let int_tok = toks.iter().find(|t| t.kind == TokenKind::Int(1)).unwrap();
        assert_eq!(int_tok.span.col_start, 5,
            "col_start should count chars, not bytes");
    }

    // ── [D2] Arc<str> – shared filename ──────────────────────────────────────

    #[test]
    fn span_file_arc_is_shared() {
        let mut l = Lexer::new("1 2", "myfile.lang");
        let (toks, _) = l.tokenize();
        let p0 = Arc::as_ptr(&toks[0].span.file);
        let p1 = Arc::as_ptr(&toks[1].span.file);
        assert_eq!(p0, p1, "all spans must share the same Arc<str>");
    }

    // ── [D3] ASCII-only identifiers ───────────────────────────────────────────

    #[test]
    fn ascii_ident_accepted() {
        let toks = lex_ok("foo_Bar123");
        assert_eq!(toks[0], TokenKind::Ident("foo_Bar123".into()));
    }

    #[test]
    fn non_ascii_ident_start_is_error() {
        // 'é' is alphabetic but NOT ascii_alphabetic.
        let errs = lex_errs("éclair");
        assert!(!errs.is_empty(), "non-ASCII ident start should produce an error");
        assert!(matches!(errs[0], LexError::UnexpectedChar { .. }));
    }

    // ── [D4] compound-assignment and ? tokens ─────────────────────────────────

    #[test]
    fn compound_assign_tokens() {
        for (src, expected) in [
            ("+=", TokenKind::PlusEq),
            ("-=", TokenKind::MinusEq),
            ("*=", TokenKind::StarEq),
            ("/=", TokenKind::SlashEq),
            ("%=", TokenKind::PercentEq),
        ] {
            assert_eq!(lex_ok(src)[0], expected, "failed for `{}`", src);
        }
    }

    #[test]
    fn question_tokens() {
        let toks = lex_ok("? ??");
        assert_eq!(toks[0], TokenKind::Question);
        assert_eq!(toks[1], TokenKind::QuestionQuestion);
    }

    // ── [D5] error recovery ───────────────────────────────────────────────────

    #[test]
    fn multiple_errors_in_one_pass() {
        let (toks, errs) = lex("let @ x = $ 1;");
        assert_eq!(errs.len(), 2, "should collect both errors, got: {:?}", errs);
        assert!(matches!(errs[0], LexError::UnexpectedChar { ch: '@', .. }));
        assert!(matches!(errs[1], LexError::UnexpectedChar { ch: '$', .. }));
        assert!(toks.iter().any(|t| t.kind == TokenKind::Let));
        assert!(toks.iter().any(|t| t.kind == TokenKind::Int(1)));
    }

    #[test]
    fn invalid_escape_recovery_continues_string() {
        let (toks, errs) = lex(r#""\q world""#);
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0], LexError::InvalidEscape { ch: 'q', .. }));
        // Token still emitted with recovered content.
        assert!(toks.iter().any(|t| matches!(&t.kind, TokenKind::Str(s) if s.contains('q'))));
    }

    #[test]
    fn unterminated_string_then_valid_token() {
        let (toks, errs) = lex("\"oops\n42");
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0], LexError::UnterminatedString { .. }));
        assert!(toks.iter().any(|t| t.kind == TokenKind::Int(42)));
    }

    #[test]
    fn bitwise_ops_recovered_valid_tokens_survive() {
        let (toks, errs) = lex("a & b | c");
        assert_eq!(errs.len(), 2);
        let idents: Vec<_> = toks.iter()
            .filter_map(|t| if let TokenKind::Ident(s) = &t.kind { Some(s.as_str()) } else { None })
            .collect();
        assert!(idents.contains(&"a") && idents.contains(&"b") && idents.contains(&"c"));
    }

    // ── [D6] nested block comments ────────────────────────────────────────────

    #[test]
    fn nested_comment_correctly_closed() {
        assert_eq!(lex_ok("/* /* inner */ */ 99")[0], TokenKind::Int(99));
    }

    #[test]
    fn nested_comment_unterminated_outer() {
        let errs = lex_errs("/* /* inner */");
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0], LexError::UnterminatedBlockComment { .. }));
    }

    #[test]
    fn triple_nested_comment() {
        assert_eq!(lex_ok("/* a /* b /* c */ b */ a */ 7")[0], TokenKind::Int(7));
    }

    #[test]
    fn nested_depth_reported_in_error() {
        // Two unclosed levels: depth should be 2.
        let errs = lex_errs("/* /* unclosed");
        assert!(matches!(errs[0], LexError::UnterminatedBlockComment { max_depth: 2, .. }));
    }

    // ── diagnostic render smoke tests ─────────────────────────────────────────

    #[test]
    fn all_error_variants_render_without_panic() {
        let cases: &[&str] = &[
            "let x = @val;",          // L001
            "\"oops",                 // L002
            "/* unclosed",            // L003
            r#""\q""#,                // L004
            "9999999999999999999999", // L005
            "a & b",                  // L006
        ];
        for src in cases {
            let (_, errs) = lex(src);
            assert!(!errs.is_empty(), "expected error for: {}", src);
            let lines: Vec<&str> = src.lines().collect();
            for e in errs {
                let out = format!("{}", Diagnostic::new(e, &lines));
                assert!(!out.is_empty(), "diagnostic produced empty output for: {}", src);
            }
        }
    }
}