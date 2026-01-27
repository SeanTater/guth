use guth::audio::io::WavIo;
use std::process::Command;

#[test]
fn audio_convert_resamples_and_converts_channels() {
    let dir = tempfile::tempdir().expect("tempdir");
    let input = dir.path().join("input.wav");
    let output = dir.path().join("output.wav");

    let samples = vec![vec![0.0_f32; 480]];
    WavIo::write_wav(&input, &samples, 48000).expect("write input wav");

    let status = Command::new(env!("CARGO_BIN_EXE_guth"))
        .args([
            "audio",
            "convert",
            "--input",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--to-rate",
            "24000",
            "--to-channels",
            "2",
        ])
        .status()
        .expect("run guth audio convert");

    assert!(status.success());
    let (converted, sample_rate) = WavIo::read_wav(&output).expect("read output wav");
    assert_eq!(sample_rate, 24000);
    assert_eq!(converted.len(), 2);
    assert!(!converted[0].is_empty());
}
