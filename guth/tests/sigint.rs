#[cfg(unix)]
mod unix {
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;
    use std::process::{Command, Stdio};
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn say_exits_on_sigint() {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_guth"));
        cmd.args([
            "say",
            "Hello there.",
            "--output",
            "/tmp/guth_sigint.wav",
            "--config",
            "tests/fixtures/tts_integration_config.yaml",
            "--max-gen-len",
            "8192",
            "--frames-after-eos",
            "0",
            "--stream",
            "--progress",
        ]);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        let child = cmd.spawn().expect("spawn guth");
        sleep(Duration::from_millis(10));
        let pid = Pid::from_raw(child.id() as i32);
        let _ = kill(pid, Signal::SIGINT);

        let output = child.wait_with_output().expect("wait output");
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stderr.contains("Interrupted") || stdout.contains("Interrupted"),
            "output did not mention interruption: stderr={stderr} stdout={stdout}"
        );
    }
}
