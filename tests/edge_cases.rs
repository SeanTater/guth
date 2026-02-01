use burn_ndarray::{NdArray, NdArrayDevice};
use guth::config::load_config;
use guth::model::tts::TtsModel;

#[test]
fn missing_weights_path_errors() {
    let config = load_config("tests/fixtures/tts_integration_config.yaml").expect("load config");
    let mut config = config.clone();
    config.weights_path = Some("tests/fixtures/missing_weights.safetensors".to_string());

    let device = NdArrayDevice::default();
    let err =
        TtsModel::<NdArray<f32>>::from_config(&config, 0.0, 2, None, 0.0, &device).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("no such file"));
}

#[test]
fn fallback_weights_disable_voice_cloning() {
    let config = load_config("tests/fixtures/tts_integration_config.yaml").expect("load config");
    let mut config = config.clone();
    config.weights_path = Some("tests/fixtures/missing_weights.safetensors".to_string());
    config.weights_path_without_voice_cloning =
        Some("tests/fixtures/tts_state.safetensors".to_string());

    let device = NdArrayDevice::default();
    let model =
        TtsModel::<NdArray<f32>>::from_config(&config, 0.0, 2, None, 0.0, &device).unwrap();
    assert!(!model.voice_cloning_supported);
}
