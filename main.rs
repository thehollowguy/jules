// ─────────────────────────────────────────────────────────────────────────────
// MAIN  — driver, REPL, and compiler pipeline
// ─────────────────────────────────────────────────────────────────────────────
mod lexer;
mod ast;
mod parser;
mod types;
mod interpreter;

use std::io::{self, Write};
use lexer::Lexer;
use parser::Parser;
use types::Analyser;
use interpreter::Interpreter;

// ── Pipeline ──────────────────────────────────────────────────────────────────

/// One full pass over source text.
/// Returns the last evaluated value (for REPL), or an error string.
fn run(source: &str, file: &str, interp: &mut Interpreter) -> Result<interpreter::Value, String> {
    // ── 1. Lex ────────────────────────────────────────────────────────────────
    let mut lexer = Lexer::new(source, file);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;

    // ── 2. Parse ──────────────────────────────────────────────────────────────
    let mut parser = Parser::new(tokens);
    let module = parser.parse_module(file);

    if !parser.errors().is_empty() {
        let msgs: Vec<String> = parser.errors().iter().map(|e| e.to_string()).collect();
        return Err(msgs.join("\n"));
    }

    // ── 3. Semantic analysis / type-check ─────────────────────────────────────
    let mut analyser = Analyser::new();
    analyser.analyse_module(&module);

    if !analyser.errors.is_empty() {
        let msgs: Vec<String> = analyser.errors.iter().map(|e| e.to_string()).collect();
        // Treat type errors as warnings for now — still run
        eprintln!("[type-check warnings]\n{}", msgs.join("\n"));
    }

    // ── 4. Interpret ──────────────────────────────────────────────────────────
    interp.run_module(&module).map_err(|e| e.to_string())
}

// ── REPL ──────────────────────────────────────────────────────────────────────

fn repl() {
    println!("╔══════════════════════════════════════╗");
    println!("║    lang_skeleton  v0.1.0  REPL       ║");
    println!("║  type :quit to exit, :help for help  ║");
    println!("╚══════════════════════════════════════╝");

    let mut interp = Interpreter::new();
    let mut input = String::new();

    loop {
        if input.is_empty() { print!(">>> "); } else { print!("... "); }
        io::stdout().flush().unwrap();

        let mut line = String::new();
        if io::stdin().read_line(&mut line).is_err() { break; }

        let trimmed = line.trim();
        match trimmed {
            ":quit" | ":q" => { println!("bye!"); break; }
            ":help"        => { print_help(); input.clear(); continue; }
            ""             => {
                if !input.is_empty() {
                    // blank line after multi-line = execute
                    let src = input.clone();
                    input.clear();
                    match run(&src, "repl", &mut interp) {
                        Ok(v)  => { if !matches!(v, interpreter::Value::Unit | interpreter::Value::Nil) { println!("= {}", v); } }
                        Err(e) => eprintln!("{}", e),
                    }
                }
                continue;
            }
            _ => {}
        }

        input.push_str(&line);

        // Heuristic: if braces are balanced, run immediately
        if braces_balanced(&input) {
            let src = input.clone();
            input.clear();
            match run(&src, "repl", &mut interp) {
                Ok(v)  => { if !matches!(v, interpreter::Value::Unit | interpreter::Value::Nil) { println!("= {}", v); } }
                Err(e) => eprintln!("{}", e),
            }
        }
    }
}

fn braces_balanced(s: &str) -> bool {
    let (mut depth, mut in_str) = (0i32, false);
    for c in s.chars() {
        if c == '"'  { in_str = !in_str; }
        if in_str    { continue; }
        if c == '{'  { depth += 1; }
        if c == '}'  { depth -= 1; }
    }
    depth == 0
}

fn print_help() {
    println!("─────────────────────────────────────────");
    println!("  Language quick-reference");
    println!("  Variables : let x = 42");
    println!("             let mut y = 0");
    println!("  Functions : fn add(a: Int, b: Int) -> Int {{ return a + b; }}");
    println!("  Structs   : struct Point {{ x: Int, y: Int }}");
    println!("  Enums     : enum Color {{ Red, Green, Blue }}");
    println!("  If/else   : if x > 0 {{ println(x) }} else {{ println(0) }}");
    println!("  While     : while x < 10 {{ x = x + 1; }}");
    println!("  For       : for i in [1, 2, 3] {{ println(i) }}");
    println!("  Match     : match x {{ 1 => println(\"one\"), _ => println(\"other\") }}");
    println!("  Lambdas   : let f = |x| x * 2");
    println!("  Builtins  : println, print, len, push, to_str, type_of, assert");
    println!("─────────────────────────────────────────");
}

// ── File runner ───────────────────────────────────────────────────────────────

fn run_file(path: &str) {
    let source = match std::fs::read_to_string(path) {
        Ok(s)  => s,
        Err(e) => { eprintln!("could not read `{}`: {}", path, e); std::process::exit(1); }
    };
    let mut interp = Interpreter::new();
    match run(&source, path, &mut interp) {
        Ok(_)  => {}
        Err(e) => { eprintln!("{}", e); std::process::exit(1); }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        None | Some("repl") => repl(),
        Some("run") => {
            match args.get(2) {
                Some(path) => run_file(path),
                None       => { eprintln!("usage: lang run <file>"); std::process::exit(1); }
            }
        }
        Some("check") => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or_else(|| {
                eprintln!("usage: lang check <file>"); std::process::exit(1);
            });
            let source = std::fs::read_to_string(path).unwrap_or_else(|e| {
                eprintln!("{}", e); std::process::exit(1);
            });
            let mut lexer = Lexer::new(&source, path);
            match lexer.tokenize() {
                Err(e) => { eprintln!("{}", e); std::process::exit(1); }
                Ok(tokens) => {
                    let mut parser = Parser::new(tokens);
                    let module = parser.parse_module(path);
                    for e in parser.errors() { eprintln!("{}", e); }
                    let mut analyser = Analyser::new();
                    analyser.analyse_module(&module);
                    for e in &analyser.errors { eprintln!("{}", e); }
                    if parser.errors().is_empty() && analyser.errors.is_empty() {
                        println!("✓ no errors found in `{}`", path);
                    } else {
                        std::process::exit(1);
                    }
                }
            }
        }
        Some("tokens") => {
            // Debug: dump token stream
            let path = args.get(2).map(|s| s.as_str()).unwrap_or_else(|| {
                eprintln!("usage: lang tokens <file>"); std::process::exit(1);
            });
            let source = std::fs::read_to_string(path).unwrap_or_default();
            let mut lexer = Lexer::new(&source, path);
            match lexer.tokenize() {
                Ok(tokens) => {
                    for t in &tokens { println!("{:?}", t); }
                }
                Err(e) => eprintln!("{}", e),
            }
        }
        Some(cmd) => {
            eprintln!("unknown command `{}`. Available: repl, run, check, tokens", cmd);
            std::process::exit(1);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BUILT-IN TESTS (run with `cargo test`)
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn eval(src: &str) -> interpreter::Value {
        let mut interp = Interpreter::new();
        run(src, "test", &mut interp).expect("eval failed")
    }

    #[test] fn test_arithmetic()  { let _ = eval("let x = 2 + 3 * 4;"); }
    #[test] fn test_let_binding() { let _ = eval("let x = 10; let y = x + 5;"); }
    #[test] fn test_fn_call()     { let _ = eval("fn double(n: Int) -> Int { return n * 2; } let r = double(7);"); }
    #[test] fn test_if_expr()     { let _ = eval("let x = if true { 1 } else { 2 };"); }
    #[test] fn test_while_loop()  { let _ = eval("let mut i = 0; while i < 5 { i = i + 1; }"); }
    #[test] fn test_for_loop()    { let _ = eval("for x in [1, 2, 3] { println(x); }"); }
    #[test] fn test_struct()      { let _ = eval("struct Point { x: Int, y: Int } let p = Point { x: 1, y: 2 };"); }
    #[test] fn test_string_concat(){ let _ = eval(r#"let s = "hello" + " world";"#); }
    #[test] fn test_array()       { let _ = eval("let a = [1, 2, 3]; let n = len(a);"); }
    #[test] fn test_match()       { let _ = eval("let x = 2; match x { 1 => println(\"one\"), _ => println(\"other\") }"); }
}
