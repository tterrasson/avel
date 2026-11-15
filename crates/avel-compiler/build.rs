use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

fn main() -> io::Result<()> {
    let source = Path::new("src/codegen/wgsl");
    let target = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR")).join("wgsl");
    println!("cargo:rerun-if-changed={}", source.display());
    if target.exists() {
        fs::remove_dir_all(&target)?;
    }
    compile_directory(source, &target)
}

fn compile_directory(source: &Path, target: &Path) -> io::Result<()> {
    fs::create_dir_all(target)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        let output = target.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            compile_directory(&path, &output)?;
        } else if path.extension().is_some_and(|ext| ext == "wgsl") {
            let source = fs::read_to_string(&path)?;
            let template = compile_template(&source).map_err(|message| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{}: {message}", path.display()),
                )
            })?;
            // Write a Rust string literal for include! in the wgsl! macro.
            fs::write(output.with_extension("wgsl.rs"), format!("{template:?}\n"))?;
        }
    }
    Ok(())
}

/// WGSL uses ordinary braces. Only `{{name}}` becomes a Rust format parameter.
fn compile_template(mut source: &str) -> Result<String, &'static str> {
    fn push_text(output: &mut String, text: &str) {
        for ch in text.chars() {
            if matches!(ch, '{' | '}') {
                output.push(ch);
            }
            output.push(ch);
        }
    }

    let mut output = String::with_capacity(source.len());
    while let Some((text, hole)) = source.split_once("{{") {
        push_text(&mut output, text);
        let (name, rest) = hole.split_once("}}").ok_or("unclosed WGSL placeholder")?;
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err("invalid WGSL placeholder name");
        }
        output.push('{');
        output.push_str(name);
        output.push('}');
        source = rest;
    }
    push_text(&mut output, source);
    Ok(output)
}
