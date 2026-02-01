use std::process::Command;

#[test]
fn cli_lists_voices() {
    let output = Command::new(env!("CARGO_BIN_EXE_guth"))
        .args(["voices"])
        .output()
        .expect("run guth voices");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("alba"));
}

#[test]
fn cli_lists_models() {
    let output = Command::new(env!("CARGO_BIN_EXE_guth"))
        .args(["models"])
        .output()
        .expect("run guth models");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("b6369a24"));
}
