use guth::audio::io::WavIo;
use safetensors::SafeTensors;
use std::path::PathBuf;
use std::process::Command;

fn should_run() -> bool {
    std::env::var("GUTH_E2E").map(|v| v == "1").unwrap_or(false)
}

fn config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("pocket_tts")
        .join("config")
        .join("b6369a24.yaml")
}

fn run(cmd: &mut Command) {
    let status = cmd.status().expect("run command");
    assert!(status.success());
}

#[test]
fn cli_say_generates_audio() {
    if !should_run() {
        eprintln!("Skipping E2E test; set GUTH_E2E=1 to enable.");
        return;
    }

    run(Command::new(env!("CARGO_BIN_EXE_guth")).args(["download", "b6369a24"]));

    let dir = tempfile::tempdir().expect("tempdir");
    let output = dir.path().join("say.wav");
    run(Command::new(env!("CARGO_BIN_EXE_guth")).args([
        "say",
        "Hello.",
        "--output",
        output.to_str().unwrap(),
        "--config",
        config_path().to_str().unwrap(),
        "--max-gen-len",
        "2",
        "--frames-after-eos",
        "0",
    ]));

    let (decoded, sample_rate) = WavIo::read_wav(&output).expect("read wav");
    assert_eq!(sample_rate, 24000);
    assert!(!decoded.is_empty());
    assert!(!decoded[0].is_empty());
}

#[test]
fn cli_say_streaming_generates_audio() {
    if !should_run() {
        eprintln!("Skipping E2E test; set GUTH_E2E=1 to enable.");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let output = dir.path().join("say_stream.wav");
    run(Command::new(env!("CARGO_BIN_EXE_guth")).args([
        "say",
        "Hello.",
        "--output",
        output.to_str().unwrap(),
        "--config",
        config_path().to_str().unwrap(),
        "--max-gen-len",
        "2",
        "--frames-after-eos",
        "0",
        "--stream",
    ]));

    let (decoded, sample_rate) = WavIo::read_wav(&output).expect("read wav");
    assert_eq!(sample_rate, 24000);
    assert!(!decoded.is_empty());
    assert!(!decoded[0].is_empty());
}

#[test]
fn cli_voice_encode_creates_safetensors() {
    if !should_run() {
        eprintln!("Skipping E2E test; set GUTH_E2E=1 to enable.");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let input = dir.path().join("voice.wav");
    let output = dir.path().join("voice.safetensors");
    let samples = vec![vec![0.0_f32; 480]];
    WavIo::write_wav(&input, &samples, 48000).expect("write voice wav");

    run(Command::new(env!("CARGO_BIN_EXE_guth")).args([
        "voice",
        "encode",
        "--input",
        input.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--config",
        config_path().to_str().unwrap(),
    ]));

    assert!(output.exists());
    assert!(output.metadata().expect("metadata").len() > 0);

    let data = std::fs::read(&output).expect("read safetensors");
    let safetensors = SafeTensors::deserialize(&data).expect("deserialize safetensors");
    let tensor = safetensors
        .tensor("audio_prompt")
        .expect("audio_prompt tensor");
    assert_eq!(tensor.shape().len(), 3);
}

#[test]
fn cli_say_accepts_voice_safetensors() {
    if !should_run() {
        eprintln!("Skipping E2E test; set GUTH_E2E=1 to enable.");
        return;
    }

    run(Command::new(env!("CARGO_BIN_EXE_guth")).args(["download", "b6369a24"]));

    let dir = tempfile::tempdir().expect("tempdir");
    let input = dir.path().join("voice.wav");
    let voice = dir.path().join("voice.safetensors");
    let output = dir.path().join("say.wav");
    let samples = vec![vec![0.0_f32; 480]];
    WavIo::write_wav(&input, &samples, 48000).expect("write voice wav");

    run(Command::new(env!("CARGO_BIN_EXE_guth")).args([
        "voice",
        "encode",
        "--input",
        input.to_str().unwrap(),
        "--output",
        voice.to_str().unwrap(),
        "--config",
        config_path().to_str().unwrap(),
    ]));

    run(Command::new(env!("CARGO_BIN_EXE_guth")).args([
        "say",
        "Hello.",
        "--output",
        output.to_str().unwrap(),
        "--config",
        config_path().to_str().unwrap(),
        "--max-gen-len",
        "2",
        "--frames-after-eos",
        "0",
        "--voice-file",
        voice.to_str().unwrap(),
    ]));

    assert!(output.exists());
}
