use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn compile(graph: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_avel"))
        .arg("compile")
        .arg(graph)
        .args(args)
        .output()
        .unwrap()
}

fn compile_identity(args: &[&str]) -> Output {
    compile(&fixture("identity.graph.json"), args)
}

#[test]
fn prints_a_bucketed_plan_with_the_requested_shape() {
    let output = compile_identity(&["--input", "x=17,4", "--bucket", "16"]);
    assert_eq!(output.status.code(), Some(0));
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(plan["outputs"][0]["shape"], serde_json::json!([32, 4]));
    assert_eq!(plan["requested"]["outputs"], serde_json::json!([[17, 4]]));
}

#[test]
fn compiles_the_mlp_fixture() {
    let output = compile(&fixture("mlp.graph.json"), &["--input", "input=3,4"]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(plan["outputs"][0]["shape"], serde_json::json!([3, 2]));
    // Four parameters, four storage buffers plus input, arena and output.
    assert!(plan["buffers"].as_array().unwrap().len() > 4);
}

#[test]
fn writes_the_plan_to_a_file_with_an_empty_stdout() {
    let path = std::env::temp_dir().join(format!("avel-cli-test-{}-plan.json", std::process::id()));
    let output = compile_identity(&["--input", "x=2,4", "--output", path.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(0));
    assert!(output.stdout.is_empty());
    let written = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let plan: serde_json::Value = serde_json::from_slice(&written).unwrap();
    assert_eq!(plan["outputs"][0]["shape"], serde_json::json!([2, 4]));
}

#[test]
fn rejects_duplicate_inputs_without_printing_a_plan() {
    let output = compile_identity(&["--input", "x=17,4", "--input", "x=20,4"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("input shape for `x` was given more than once")
    );
}

#[test]
fn rejects_malformed_input_arguments_with_a_usage_error() {
    for (arg, message) in [
        ("x=abc", "invalid dimension `abc`"),
        ("x", "expected name=shape"),
        ("=1,2", "input name must not be empty"),
    ] {
        let output = compile_identity(&["--input", arg]);
        assert_eq!(output.status.code(), Some(2), "{arg}");
        assert!(output.stdout.is_empty(), "{arg}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(message),
            "{arg}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn rejects_a_missing_graph_file() {
    let output = compile(&fixture("does-not-exist.graph.json"), &["--input", "x=2,4"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(!output.stderr.is_empty());
}

#[test]
fn rejects_a_graph_that_needs_a_missing_input_shape() {
    let output = compile_identity(&[]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("MissingInputShape"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
