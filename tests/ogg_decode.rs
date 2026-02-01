use serde::Deserialize;

use guth::audio::io::WavIo;

#[derive(Debug, Deserialize)]
struct OggFixture {
    sample_rate: u32,
    channels: usize,
    samples: Vec<f32>,
}

fn read_fixture<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = format!("tests/fixtures/{name}");
    let data = std::fs::read_to_string(path).expect("fixture read");
    serde_json::from_str(&data).expect("fixture parse")
}

#[test]
fn ogg_decode_matches_fixture() {
    let fixture: OggFixture = read_fixture("voice_ogg_fixture.json");
    let (samples, sample_rate) =
        WavIo::read_ogg("tests/fixtures/voices/sean.ogg").expect("read ogg");
    assert_eq!(sample_rate, fixture.sample_rate);
    assert_eq!(samples.len(), fixture.channels);

    let expected = &fixture.samples;
    let decoded = &samples[0][..expected.len()];
    for (idx, (a, b)) in decoded.iter().zip(expected.iter()).enumerate() {
        if (a - b).abs() > 1e-4 {
            panic!("mismatch at {idx}: {a} vs {b}");
        }
    }
}
