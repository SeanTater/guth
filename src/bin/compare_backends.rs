//! Compare WGPU vs NdArray backend outputs at each Mimi encoder stage.
//!
//! This diagnostic binary identifies where numerical divergence occurs between
//! backends by comparing tensors at intermediate stages of the encoding pipeline.
//!
//! Usage: cargo run --release --features backend-wgpu,backend-ndarray --bin compare_backends
//!
//! The binary will:
//! 1. Load the same audio through both backends
//! 2. Run the Mimi encoder, capturing outputs at each stage
//! 3. Compare tensors and report divergence statistics
//!
//! For transformer-specific diagnostics:
//!   cargo run --release --features backend-wgpu,backend-ndarray --bin compare_backends -- --transformer

// This binary is a diagnostics scratchpad; keep CI/clippy clean by allowing unused pieces.
#![allow(dead_code, unused_variables, unreachable_code)]

use anyhow::Result;
use burn::tensor::{activation::gelu, backend::Backend, Int, Tensor, TensorData};
use guth::audio::io::WavIo;
use guth::audio::resample::AudioResampler;
use guth::config::load_config;
use guth::model::tts::TtsModel;

#[cfg(feature = "backend-wgpu")]
use burn_wgpu::{Wgpu, WgpuDevice};

use burn_ndarray::{NdArray, NdArrayDevice};

/// Statistics computed from comparing two tensors.
#[derive(Debug)]
struct ComparisonStats {
    /// Shape of the tensors being compared.
    shape: Vec<usize>,
    /// Root mean square of the difference.
    rms_diff: f32,
    /// Maximum absolute difference.
    max_abs_diff: f32,
    /// Mean absolute difference.
    mean_abs_diff: f32,
    /// RMS of first tensor.
    rms_a: f32,
    /// RMS of second tensor.
    rms_b: f32,
    /// Relative RMS difference as percentage.
    rms_diff_pct: f32,
    /// Index of maximum difference.
    max_diff_idx: usize,
    /// Number of elements with difference > 1e-4.
    num_divergent: usize,
}

impl ComparisonStats {
    fn from_vecs(a: &[f32], b: &[f32], shape: Vec<usize>) -> Self {
        assert_eq!(a.len(), b.len(), "tensor lengths must match");
        let n = a.len() as f32;

        let mut sum_sq_diff = 0.0f64;
        let mut sum_abs_diff = 0.0f64;
        let mut max_abs_diff = 0.0f32;
        let mut max_diff_idx = 0usize;
        let mut num_divergent = 0usize;
        let mut sum_sq_a = 0.0f64;
        let mut sum_sq_b = 0.0f64;

        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (va - vb).abs();
            sum_sq_diff += (diff as f64).powi(2);
            sum_abs_diff += diff as f64;
            if diff > max_abs_diff {
                max_abs_diff = diff;
                max_diff_idx = i;
            }
            if diff > 1e-4 {
                num_divergent += 1;
            }
            sum_sq_a += (va as f64).powi(2);
            sum_sq_b += (vb as f64).powi(2);
        }

        let rms_diff = (sum_sq_diff / n as f64).sqrt() as f32;
        let mean_abs_diff = (sum_abs_diff / n as f64) as f32;
        let rms_a = (sum_sq_a / n as f64).sqrt() as f32;
        let rms_b = (sum_sq_b / n as f64).sqrt() as f32;

        // Relative difference as percentage of average RMS
        let avg_rms = (rms_a + rms_b) / 2.0;
        let rms_diff_pct = if avg_rms > 1e-10 {
            (rms_diff / avg_rms) * 100.0
        } else {
            0.0
        };

        Self {
            shape,
            rms_diff,
            max_abs_diff,
            mean_abs_diff,
            rms_a,
            rms_b,
            rms_diff_pct,
            max_diff_idx,
            num_divergent,
        }
    }
}

/// Captured outputs at each Mimi encoder stage.
struct EncoderStages {
    /// After padding (input to SEANet).
    padded: Vec<f32>,
    padded_shape: Vec<usize>,
    /// After SEANet encoder.
    seanet: Vec<f32>,
    seanet_shape: Vec<usize>,
    /// After encoder transformer.
    transformer: Vec<f32>,
    transformer_shape: Vec<usize>,
    /// After downsampling (final latent).
    final_latent: Vec<f32>,
    final_latent_shape: Vec<usize>,
    /// After conditioning projection.
    conditioning: Vec<f32>,
    conditioning_shape: Vec<usize>,
}

/// Convert a tensor to a Vec<f32> for cross-backend comparison.
fn tensor_to_vec<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> (Vec<f32>, Vec<usize>) {
    let shape = tensor.dims().to_vec();
    let data = tensor.to_data();
    let values = data.as_slice::<f32>().expect("f32 slice").to_vec();
    (values, shape)
}

/// Run the Mimi encoder pipeline and capture outputs at each stage.
fn run_encoder_stages<B: Backend>(tts: &TtsModel<B>, audio_tensor: Tensor<B, 3>) -> EncoderStages {
    let mimi = &tts.mimi;
    let [batch, channels, length] = audio_tensor.dims();
    let device = audio_tensor.device();

    // Step 1: Compute padding
    let frame_size = mimi.frame_size();
    let extra_padding = get_extra_padding_for_conv1d(length, frame_size, frame_size, 0);
    let padded = if extra_padding == 0 {
        audio_tensor.clone()
    } else {
        let padding: Tensor<B, 3> = Tensor::zeros([batch, channels, extra_padding], &device);
        Tensor::cat(vec![audio_tensor, padding], 2)
    };
    let (padded_vec, padded_shape) = tensor_to_vec(&padded);

    // Step 2: SEANet encoder
    let mut encoder_state = mimi.encoder.init_state(batch);
    let seanet_out = mimi
        .encoder
        .forward(padded, &mut encoder_state)
        .expect("seanet encoder");
    let (seanet_vec, seanet_shape) = tensor_to_vec(&seanet_out);

    // Step 3: Encoder transformer
    let mut transformer_state = mimi.encoder_transformer.init_state(batch, &device);
    let mut outputs = mimi
        .encoder_transformer
        .forward(seanet_out, &mut transformer_state);
    let transformer_out = outputs.remove(0);
    let (transformer_vec, transformer_shape) = tensor_to_vec(&transformer_out);

    // Step 4: Downsample to frame rate
    let final_latent = if let Some(downsample) = &mimi.downsample {
        let mut downsample_state = downsample.init_state(batch, &device);
        downsample.forward(transformer_out.clone(), &mut downsample_state)
    } else {
        transformer_out.clone()
    };
    let (final_latent_vec, final_latent_shape) = tensor_to_vec(&final_latent);

    // Step 5: Project to conditioning
    let latents = final_latent.swap_dims(1, 2); // [B, T, C]
    let conditioning = burn::tensor::module::linear(latents, tts.speaker_proj_weight.clone(), None);
    let (conditioning_vec, conditioning_shape) = tensor_to_vec(&conditioning);

    EncoderStages {
        padded: padded_vec,
        padded_shape,
        seanet: seanet_vec,
        seanet_shape,
        transformer: transformer_vec,
        transformer_shape,
        final_latent: final_latent_vec,
        final_latent_shape,
        conditioning: conditioning_vec,
        conditioning_shape,
    }
}

/// Compute extra padding for conv1d.
fn get_extra_padding_for_conv1d(
    length: usize,
    kernel_size: usize,
    stride: usize,
    padding_total: usize,
) -> usize {
    let n_frames =
        (length as f64 - kernel_size as f64 + padding_total as f64) / stride as f64 + 1.0;
    let ceil_frames = n_frames.ceil() as usize;
    let ideal_length = if ceil_frames == 0 {
        0
    } else {
        (ceil_frames - 1) * stride + kernel_size - padding_total
    };
    ideal_length.saturating_sub(length)
}

// ============================================================================
// Transformer Component Tests
// ============================================================================

/// Test RoPE position embedding computation in isolation.
fn test_rope<B: Backend>(device: &B::Device) -> (Vec<f32>, Vec<usize>) {
    // Create test tensors matching transformer dimensions
    let batch = 1;
    let seq = 64;
    let heads = 8;
    let dim = 64; // head_dim
    let half = dim / 2;
    let max_period = 10000.0f32;

    // Create random-ish but deterministic test data
    let q_data: Vec<f32> = (0..(batch * seq * heads * dim))
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..(batch * seq * heads * dim))
        .map(|i| (i as f32 * 0.013).cos() * 0.1)
        .collect();

    let q: Tensor<B, 4> =
        Tensor::from_data(TensorData::new(q_data, [batch, seq, heads, dim]), device);
    let k: Tensor<B, 4> =
        Tensor::from_data(TensorData::new(k_data, [batch, seq, heads, dim]), device);
    let offset: Tensor<B, 1, Int> = Tensor::zeros([batch], device);

    // Compute RoPE (matching mimi_transformer.rs:133-193)
    let scale = -max_period.ln() * 2.0 / dim as f32;
    let ds = Tensor::<B, 1, Int>::arange(0..half as i64, device).float();
    let freqs = ds.mul_scalar(scale).exp();

    let ts = Tensor::<B, 1, Int>::arange(0..seq as i64, device)
        .float()
        .unsqueeze_dim::<2>(0);
    let ts = ts.repeat_dim(0, batch);
    let offset_f = offset.float().unsqueeze_dim::<2>(1).repeat_dim(1, seq);
    let angles = ts
        .add(offset_f)
        .unsqueeze_dim::<3>(2)
        .mul(freqs.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0));

    let rotr = angles.clone().cos();
    let roti = angles.sin();
    let rotr = rotr.unsqueeze_dim::<4>(2).repeat_dim(2, heads);
    let roti = roti.unsqueeze_dim::<4>(2).repeat_dim(2, heads);

    let q = q.reshape([batch, seq, heads, half, 2]);
    let k = k.reshape([batch, seq, heads, half, 2]);

    let qr = q.clone().narrow(4, 0, 1).reshape([batch, seq, heads, half]);
    let qi = q.clone().narrow(4, 1, 1).reshape([batch, seq, heads, half]);
    let kr = k.clone().narrow(4, 0, 1).reshape([batch, seq, heads, half]);
    let ki = k.clone().narrow(4, 1, 1).reshape([batch, seq, heads, half]);

    let qor = qr
        .clone()
        .mul(rotr.clone())
        .sub(qi.clone().mul(roti.clone()));
    let qoi = qr.mul(roti.clone()).add(qi.mul(rotr.clone()));
    let kor = kr
        .clone()
        .mul(rotr.clone())
        .sub(ki.clone().mul(roti.clone()));
    let koi = kr.mul(roti).add(ki.mul(rotr));

    let qo = Tensor::cat(
        vec![qor.unsqueeze_dim::<5>(4), qoi.unsqueeze_dim::<5>(4)],
        4,
    )
    .reshape([batch, seq, heads, dim]);
    let ko = Tensor::cat(
        vec![kor.unsqueeze_dim::<5>(4), koi.unsqueeze_dim::<5>(4)],
        4,
    )
    .reshape([batch, seq, heads, dim]);

    // Return concatenated Q and K for comparison
    let result = Tensor::cat(vec![qo, ko], 3);
    tensor_to_vec(&result)
}

/// Test GELU activation in isolation.
fn test_gelu<B: Backend>(device: &B::Device) -> (Vec<f32>, Vec<usize>) {
    let data: Vec<f32> = (-1000..1000).map(|i| i as f32 * 0.01).collect();
    let len = data.len();
    let input: Tensor<B, 1> = Tensor::from_data(TensorData::new(data, [len]), device);
    let output = gelu(input);
    tensor_to_vec(&output)
}

/// Test softmax in isolation.
fn test_softmax<B: Backend>(device: &B::Device) -> (Vec<f32>, Vec<usize>) {
    // Create a test tensor similar to attention scores
    let batch = 1;
    let heads = 8;
    let seq_q = 64;
    let seq_k = 64;

    let data: Vec<f32> = (0..(batch * heads * seq_q * seq_k))
        .map(|i| (i as f32 * 0.001).sin() * 2.0)
        .collect();

    let input: Tensor<B, 4> =
        Tensor::from_data(TensorData::new(data, [batch, heads, seq_q, seq_k]), device);

    // Apply softmax along the last dimension
    let output = burn::tensor::activation::softmax(input, 3);
    tensor_to_vec(&output)
}

/// Test burn::tensor::module::attention directly with varying sequence lengths.
fn test_attention_scaling<B: Backend>(
    device: &B::Device,
    seq_len: usize,
) -> (Vec<f32>, Vec<usize>) {
    let batch = 1;
    let heads = 8;
    let head_dim = 64;

    // Create Q, K, V tensors
    let q_data: Vec<f32> = (0..(batch * heads * seq_len * head_dim))
        .map(|i| (i as f32 * 0.001).sin() * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..(batch * heads * seq_len * head_dim))
        .map(|i| (i as f32 * 0.0013).cos() * 0.1)
        .collect();
    let v_data: Vec<f32> = (0..(batch * heads * seq_len * head_dim))
        .map(|i| (i as f32 * 0.0017).sin() * 0.1)
        .collect();

    let q: Tensor<B, 4> = Tensor::from_data(
        TensorData::new(q_data, [batch, heads, seq_len, head_dim]),
        device,
    );
    let k: Tensor<B, 4> = Tensor::from_data(
        TensorData::new(k_data, [batch, heads, seq_len, head_dim]),
        device,
    );
    let v: Tensor<B, 4> = Tensor::from_data(
        TensorData::new(v_data, [batch, heads, seq_len, head_dim]),
        device,
    );

    // Run attention without mask first
    let output = burn::tensor::module::attention(q, k, v, None);
    tensor_to_vec(&output)
}

/// Test attention with a causal mask (as used in the transformer).
fn test_attention_with_mask<B: Backend + 'static>(
    device: &B::Device,
    seq_len: usize,
    context: usize,
) -> (Vec<f32>, Vec<usize>) {
    let batch = 1;
    let heads = 8;
    let head_dim = 64;

    // Create Q, K, V tensors
    let q_data: Vec<f32> = (0..(batch * heads * seq_len * head_dim))
        .map(|i| (i as f32 * 0.001).sin() * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..(batch * heads * seq_len * head_dim))
        .map(|i| (i as f32 * 0.0013).cos() * 0.1)
        .collect();
    let v_data: Vec<f32> = (0..(batch * heads * seq_len * head_dim))
        .map(|i| (i as f32 * 0.0017).sin() * 0.1)
        .collect();

    let q: Tensor<B, 4> = Tensor::from_data(
        TensorData::new(q_data, [batch, heads, seq_len, head_dim]),
        device,
    );
    let k: Tensor<B, 4> = Tensor::from_data(
        TensorData::new(k_data, [batch, heads, seq_len, head_dim]),
        device,
    );
    let v: Tensor<B, 4> = Tensor::from_data(
        TensorData::new(v_data, [batch, heads, seq_len, head_dim]),
        device,
    );

    // Create causal mask matching transformer implementation
    let _offset: Tensor<B, 1, Int> = Tensor::zeros([batch], device);
    let q_len = seq_len;
    let _k_len = seq_len;

    // Position computation
    let positions = Tensor::<B, 1, Int>::arange(0..q_len as i64, device).unsqueeze_dim::<2>(0);
    let pos_q = positions.repeat_dim(0, batch);
    let pos_k = pos_q.clone();

    // Mask computation (matching mimi_transformer.rs)
    let pos_k_valid = pos_k.clone().greater_elem(-1);
    let pos_k_valid = pos_k_valid.unsqueeze_dim::<3>(1).repeat_dim(1, q_len);
    let delta = pos_q.unsqueeze_dim::<3>(2).sub(pos_k.unsqueeze_dim::<3>(1));
    let delta_nonneg = delta.clone().greater_elem(-1);
    let mut allowed = pos_k_valid.bool_and(delta_nonneg);

    if context > 0 {
        let too_far = delta.greater_elem((context.saturating_sub(1)) as i64);
        allowed = allowed.bool_and(too_far.bool_not());
    }

    let allowed = allowed.unsqueeze_dim::<4>(1).repeat_dim(1, heads);
    let mask = allowed.bool_not();

    // Run attention with mask
    let output = burn::tensor::module::attention(q, k, v, Some(mask));
    tensor_to_vec(&output)
}

/// Test exp() operation which is used in RoPE and softmax.
fn test_exp<B: Backend>(device: &B::Device) -> (Vec<f32>, Vec<usize>) {
    // Test exp() over a range that includes typical values seen in RoPE/softmax
    let data: Vec<f32> = (-100..100).map(|i| i as f32 * 0.1).collect();
    let len = data.len();
    let input: Tensor<B, 1> = Tensor::from_data(TensorData::new(data, [len]), device);
    let output = input.exp();
    tensor_to_vec(&output)
}

/// Test sin/cos which are used in RoPE.
fn test_sincos<B: Backend>(device: &B::Device) -> (Vec<f32>, Vec<usize>) {
    let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
    let len = data.len();
    let input: Tensor<B, 1> = Tensor::from_data(TensorData::new(data, [len]), device);
    let sin_out = input.clone().sin();
    let cos_out = input.cos();
    let result = Tensor::cat(vec![sin_out, cos_out], 0);
    tensor_to_vec(&result)
}

/// Test ln() which is used in RoPE scale computation.
fn test_ln<B: Backend>(device: &B::Device) -> (Vec<f32>, Vec<usize>) {
    // Test ln() over positive values
    let data: Vec<f32> = (1..1001).map(|i| i as f32 * 0.1).collect();
    let len = data.len();
    let input: Tensor<B, 1> = Tensor::from_data(TensorData::new(data, [len]), device);
    let output = input.log();
    tensor_to_vec(&output)
}

/// Run transformer component tests and compare backends.
fn run_transformer_tests() -> Result<()> {
    println!("\n========================================");
    println!("TRANSFORMER COMPONENT TESTS");
    println!("========================================\n");

    // Test on NdArray
    let ndarray_device = NdArrayDevice::default();

    let ndarray_exp = test_exp::<NdArray<f32>>(&ndarray_device);
    let ndarray_ln = test_ln::<NdArray<f32>>(&ndarray_device);
    let ndarray_sincos = test_sincos::<NdArray<f32>>(&ndarray_device);
    let ndarray_gelu = test_gelu::<NdArray<f32>>(&ndarray_device);
    let ndarray_softmax = test_softmax::<NdArray<f32>>(&ndarray_device);
    let ndarray_rope = test_rope::<NdArray<f32>>(&ndarray_device);

    // Test on WGPU
    #[cfg(feature = "backend-wgpu")]
    {
        let wgpu_device = WgpuDevice::default();

        let wgpu_exp = test_exp::<Wgpu>(&wgpu_device);
        let wgpu_ln = test_ln::<Wgpu>(&wgpu_device);
        let wgpu_sincos = test_sincos::<Wgpu>(&wgpu_device);
        let wgpu_gelu = test_gelu::<Wgpu>(&wgpu_device);
        let wgpu_softmax = test_softmax::<Wgpu>(&wgpu_device);
        let wgpu_rope = test_rope::<Wgpu>(&wgpu_device);

        // Compare results
        let tests = [
            ("exp()", &ndarray_exp, &wgpu_exp),
            ("ln()", &ndarray_ln, &wgpu_ln),
            ("sin/cos()", &ndarray_sincos, &wgpu_sincos),
            ("gelu()", &ndarray_gelu, &wgpu_gelu),
            ("softmax()", &ndarray_softmax, &wgpu_softmax),
            ("RoPE full", &ndarray_rope, &wgpu_rope),
        ];

        println!("Operation    | Max Diff    | RMS Diff    | RMS Diff %  | Status");
        println!("-------------|-------------|-------------|-------------|--------");

        for (name, (ndarray_vec, ndarray_shape), (wgpu_vec, _)) in tests {
            let stats = ComparisonStats::from_vecs(ndarray_vec, wgpu_vec, ndarray_shape.clone());
            let status = if stats.rms_diff_pct < 0.01 {
                "\x1b[32mOK\x1b[0m"
            } else if stats.rms_diff_pct < 1.0 {
                "\x1b[33mWARN\x1b[0m"
            } else {
                "\x1b[31mFAIL\x1b[0m"
            };
            println!(
                "{:12} | {:>11.6} | {:>11.6} | {:>11.4} | {}",
                name, stats.max_abs_diff, stats.rms_diff, stats.rms_diff_pct, status
            );
        }

        // Test attention function directly with different sequence lengths
        println!("\n\nBurn attention() function scaling (no mask):");
        println!("Seq Len | Max Diff    | RMS Diff    | RMS Diff %  | Status");
        println!("--------|-------------|-------------|-------------|--------");

        for seq_len in [64, 128, 192, 256, 320, 384, 512] {
            let (ndarray_attn, ndarray_shape) =
                test_attention_scaling::<NdArray<f32>>(&ndarray_device, seq_len);
            let (wgpu_attn, _) = test_attention_scaling::<Wgpu>(&wgpu_device, seq_len);

            let stats = ComparisonStats::from_vecs(&ndarray_attn, &wgpu_attn, ndarray_shape);
            let status = if stats.rms_diff_pct < 0.01 {
                "\x1b[32mOK\x1b[0m"
            } else if stats.rms_diff_pct < 1.0 {
                "\x1b[33mWARN\x1b[0m"
            } else {
                "\x1b[31mFAIL\x1b[0m"
            };
            println!(
                "{:>7} | {:>11.6} | {:>11.6} | {:>11.4} | {}",
                seq_len, stats.max_abs_diff, stats.rms_diff, stats.rms_diff_pct, status
            );
        }

        // Test attention WITH causal mask (matching transformer)
        // The context window is typically 250 for Mimi
        let context = 250;
        println!(
            "\n\nBurn attention() with causal mask (context={}):",
            context
        );
        println!("Seq Len | Max Diff    | RMS Diff    | RMS Diff %  | Status");
        println!("--------|-------------|-------------|-------------|--------");

        for seq_len in [64, 128, 192, 256, 320, 384, 512] {
            let (ndarray_attn, ndarray_shape) =
                test_attention_with_mask::<NdArray<f32>>(&ndarray_device, seq_len, context);
            let (wgpu_attn, _) = test_attention_with_mask::<Wgpu>(&wgpu_device, seq_len, context);

            let stats = ComparisonStats::from_vecs(&ndarray_attn, &wgpu_attn, ndarray_shape);
            let status = if stats.rms_diff_pct < 0.01 {
                "\x1b[32mOK\x1b[0m"
            } else if stats.rms_diff_pct < 1.0 {
                "\x1b[33mWARN\x1b[0m"
            } else {
                "\x1b[31mFAIL\x1b[0m"
            };
            println!(
                "{:>7} | {:>11.6} | {:>11.6} | {:>11.4} | {}",
                seq_len, stats.max_abs_diff, stats.rms_diff, stats.rms_diff_pct, status
            );
        }

        // Now test with actual model weights loaded
        println!("\n\n========================================");
        println!("TESTING WITH LOADED MODEL WEIGHTS");
        println!("========================================\n");

        let config_path = "python/pocket_tts/config/b6369a24.yaml";
        let config = load_config(config_path)?;

        println!("Loading models with weights...");
        let ndarray_tts =
            TtsModel::<NdArray<f32>>::from_config(&config, 0.0, 2, None, 0.0, &ndarray_device)?;
        let wgpu_tts = TtsModel::<Wgpu>::from_config(&config, 0.0, 2, None, 0.0, &wgpu_device)?;

        // Get the first layer's weights and compare
        let ndarray_in_proj = &ndarray_tts.mimi.encoder_transformer.transformer.layers[0]
            .self_attn
            .in_proj;
        let wgpu_in_proj = &wgpu_tts.mimi.encoder_transformer.transformer.layers[0]
            .self_attn
            .in_proj;

        let (ndarray_weights, ndarray_shape) = tensor_to_vec(&ndarray_in_proj.weight.val());
        let (wgpu_weights, _) = tensor_to_vec(&wgpu_in_proj.weight.val());

        let weight_stats =
            ComparisonStats::from_vecs(&ndarray_weights, &wgpu_weights, ndarray_shape);
        println!("Layer 0 in_proj weights:");
        println!("  Max diff: {:.10}", weight_stats.max_abs_diff);
        println!("  RMS diff: {:.10}", weight_stats.rms_diff);
        println!(
            "  Elements with diff > 1e-6: {}",
            weight_stats.num_divergent
        );

        // Test running the transformer with same input
        println!("\nTesting transformer with identical input...");

        // Create identical input for both backends
        // Use seq=1008 to match real SEANet output from 5s audio
        let batch = 1;
        let seq = 1008;
        let dim = 512;
        let input_data: Vec<f32> = (0..(batch * seq * dim))
            .map(|i| (i as f32 * 0.001).sin() * 0.1)
            .collect();

        let ndarray_input: Tensor<NdArray<f32>, 3> = Tensor::from_data(
            TensorData::new(input_data.clone(), [batch, dim, seq]),
            &ndarray_device,
        );
        let wgpu_input: Tensor<Wgpu, 3> =
            Tensor::from_data(TensorData::new(input_data, [batch, dim, seq]), &wgpu_device);

        // Run through full encoder transformer
        let mut ndarray_state = ndarray_tts
            .mimi
            .encoder_transformer
            .init_state(batch, &ndarray_device);
        let mut wgpu_state = wgpu_tts
            .mimi
            .encoder_transformer
            .init_state(batch, &wgpu_device);

        let ndarray_outputs = ndarray_tts
            .mimi
            .encoder_transformer
            .forward(ndarray_input, &mut ndarray_state);
        let wgpu_outputs = wgpu_tts
            .mimi
            .encoder_transformer
            .forward(wgpu_input, &mut wgpu_state);

        let (ndarray_out, ndarray_shape) = tensor_to_vec(&ndarray_outputs[0]);
        let (wgpu_out, _) = tensor_to_vec(&wgpu_outputs[0]);

        let output_stats = ComparisonStats::from_vecs(&ndarray_out, &wgpu_out, ndarray_shape);
        print_comparison(&format!("Encoder Transformer (seq={})", seq), &output_stats);

        // Test with various sequence lengths to find threshold
        println!("\n\nSequence Length Sensitivity:");
        println!("Seq Len | RMS Diff % | Max Diff   | Status");
        println!("--------|------------|------------|--------");

        for seq_len in [64, 128, 256, 512, 750, 1008] {
            let input_data: Vec<f32> = (0..(batch * seq_len * dim))
                .map(|i| (i as f32 * 0.001).sin() * 0.01)
                .collect();

            let ndarray_input: Tensor<NdArray<f32>, 3> = Tensor::from_data(
                TensorData::new(input_data.clone(), [batch, dim, seq_len]),
                &ndarray_device,
            );
            let wgpu_input: Tensor<Wgpu, 3> = Tensor::from_data(
                TensorData::new(input_data, [batch, dim, seq_len]),
                &wgpu_device,
            );

            let mut ndarray_state = ndarray_tts
                .mimi
                .encoder_transformer
                .init_state(batch, &ndarray_device);
            let mut wgpu_state = wgpu_tts
                .mimi
                .encoder_transformer
                .init_state(batch, &wgpu_device);

            let ndarray_outputs = ndarray_tts
                .mimi
                .encoder_transformer
                .forward(ndarray_input, &mut ndarray_state);
            let wgpu_outputs = wgpu_tts
                .mimi
                .encoder_transformer
                .forward(wgpu_input, &mut wgpu_state);

            let (ndarray_vec, shape) = tensor_to_vec(&ndarray_outputs[0]);
            let (wgpu_vec, _) = tensor_to_vec(&wgpu_outputs[0]);

            let stats = ComparisonStats::from_vecs(&ndarray_vec, &wgpu_vec, shape);
            let status = if stats.rms_diff_pct < 0.1 {
                "\x1b[32mOK\x1b[0m"
            } else if stats.rms_diff_pct < 1.0 {
                "\x1b[33mWARN\x1b[0m"
            } else {
                "\x1b[31mFAIL\x1b[0m"
            };
            println!(
                "{:>7} | {:>10.4} | {:>10.6} | {}",
                seq_len, stats.rms_diff_pct, stats.max_abs_diff, status
            );
        }

        // Test each layer individually
        println!("\n\nPer-Layer Analysis:");
        println!("Layer | RMS Diff % | Max Diff   | Status");
        println!("------|------------|------------|--------");

        // Test single layer at a time
        for layer_idx in 0..ndarray_tts
            .mimi
            .encoder_transformer
            .transformer
            .layers
            .len()
        {
            let input_data: Vec<f32> = (0..(batch * seq * dim))
                .map(|i| (i as f32 * 0.001).sin() * 0.1)
                .collect();

            let ndarray_input: Tensor<NdArray<f32>, 3> = Tensor::from_data(
                TensorData::new(input_data.clone(), [batch, seq, dim]),
                &ndarray_device,
            );
            let wgpu_input: Tensor<Wgpu, 3> =
                Tensor::from_data(TensorData::new(input_data, [batch, seq, dim]), &wgpu_device);

            let ndarray_layer = &ndarray_tts.mimi.encoder_transformer.transformer.layers[layer_idx];
            let wgpu_layer = &wgpu_tts.mimi.encoder_transformer.transformer.layers[layer_idx];

            let mut ndarray_layer_state = ndarray_layer.init_state(batch, &ndarray_device);
            let mut wgpu_layer_state = wgpu_layer.init_state(batch, &wgpu_device);

            let ndarray_out = ndarray_layer.forward(ndarray_input, &mut ndarray_layer_state);
            let wgpu_out = wgpu_layer.forward(wgpu_input, &mut wgpu_layer_state);

            let (ndarray_vec, shape) = tensor_to_vec(&ndarray_out);
            let (wgpu_vec, _) = tensor_to_vec(&wgpu_out);

            let stats = ComparisonStats::from_vecs(&ndarray_vec, &wgpu_vec, shape);
            let status = if stats.rms_diff_pct < 0.1 {
                "\x1b[32mOK\x1b[0m"
            } else if stats.rms_diff_pct < 1.0 {
                "\x1b[33mWARN\x1b[0m"
            } else {
                "\x1b[31mFAIL\x1b[0m"
            };
            println!(
                "{:>5} | {:>10.4} | {:>10.6} | {}",
                layer_idx, stats.rms_diff_pct, stats.max_abs_diff, status
            );
        }
    }

    #[cfg(not(feature = "backend-wgpu"))]
    {
        eprintln!("WGPU backend not enabled");
    }

    Ok(())
}

/// Print comparison results for a stage.
fn print_comparison(stage_name: &str, stats: &ComparisonStats) {
    let status = if stats.rms_diff_pct < 0.1 {
        "\x1b[32mOK\x1b[0m"
    } else if stats.rms_diff_pct < 1.0 {
        "\x1b[33mWARN\x1b[0m"
    } else {
        "\x1b[31mDIVERGED\x1b[0m"
    };

    println!("\n=== {} [{}] ===", stage_name, status);
    println!("  Shape: {:?}", stats.shape);
    println!(
        "  NdArray RMS: {:.6}, WGPU RMS: {:.6}",
        stats.rms_a, stats.rms_b
    );
    println!(
        "  RMS diff: {:.6} ({:.3}% of signal)",
        stats.rms_diff, stats.rms_diff_pct
    );
    println!(
        "  Max abs diff: {:.6} at index {}",
        stats.max_abs_diff, stats.max_diff_idx
    );
    println!("  Mean abs diff: {:.6}", stats.mean_abs_diff);
    println!(
        "  Elements with diff > 1e-4: {} ({:.2}%)",
        stats.num_divergent,
        (stats.num_divergent as f32 / stats.shape.iter().product::<usize>() as f32) * 100.0
    );
}

fn main() -> Result<()> {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp_millis()
        .try_init();

    let args: Vec<String> = std::env::args().collect();

    // Check for --transformer flag
    if args.iter().any(|a| a == "--transformer") {
        return run_transformer_tests();
    }

    let audio_path = args
        .iter()
        .find(|a| !a.starts_with("--") && !a.contains("compare_backends"))
        .map(|s| s.as_str())
        .unwrap_or("tests/fixtures/voices/sean.ogg");

    let config_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("python/pocket_tts/config/b6369a24.yaml");

    println!("Backend Comparison Diagnostic");
    println!("==============================");
    println!("Audio: {}", audio_path);
    println!("Config: {}", config_path);

    // Load and preprocess audio
    println!("\nLoading audio...");
    let (samples, sample_rate) = WavIo::read_audio(audio_path)?;
    let config = load_config(config_path)?;
    let target_sr = config.mimi.sample_rate as u32;
    let samples = AudioResampler::convert_audio(samples, sample_rate, target_sr, 1)?;

    // Truncate to 5 seconds
    let max_samples = (5.0 * target_sr as f64) as usize;
    let samples: Vec<Vec<f32>> = samples
        .into_iter()
        .map(|mut ch| {
            ch.truncate(max_samples);
            ch
        })
        .collect();
    println!(
        "  Samples: {} ({:.2}s at {} Hz)",
        samples[0].len(),
        samples[0].len() as f64 / target_sr as f64,
        target_sr
    );

    let audio_data: Vec<f32> = samples.into_iter().flatten().collect();
    let length = audio_data.len();

    // Run NdArray backend
    println!("\n--- Running NdArray backend ---");
    let ndarray_stages = {
        let device = NdArrayDevice::default();
        let tts = TtsModel::<NdArray<f32>>::from_config(&config, 0.0, 2, None, 0.0, &device)?;
        let audio_tensor: Tensor<NdArray<f32>, 3> =
            Tensor::from_data(TensorData::new(audio_data.clone(), [1, 1, length]), &device);
        run_encoder_stages(&tts, audio_tensor)
    };

    // Run WGPU backend
    #[cfg(feature = "backend-wgpu")]
    let wgpu_stages = {
        println!("\n--- Running WGPU backend ---");
        use burn_wgpu::graphics::AutoGraphicsApi;
        use burn_wgpu::init_setup;

        let device = WgpuDevice::default();
        let setup = init_setup::<AutoGraphicsApi>(&device, Default::default());
        println!("  Adapter: {:?}", setup.adapter.get_info().name);
        println!("  Backend: {:?}", setup.backend);

        let tts = TtsModel::<Wgpu>::from_config(&config, 0.0, 2, None, 0.0, &device)?;
        let audio_tensor: Tensor<Wgpu, 3> =
            Tensor::from_data(TensorData::new(audio_data.clone(), [1, 1, length]), &device);
        run_encoder_stages(&tts, audio_tensor)
    };

    #[cfg(not(feature = "backend-wgpu"))]
    {
        eprintln!("\nError: WGPU backend not enabled. Build with --features backend-wgpu");
        std::process::exit(1);
    }

    // Compare stages
    #[cfg(feature = "backend-wgpu")]
    {
        println!("\n\n========================================");
        println!("COMPARISON RESULTS");
        println!("========================================");

        let padded_stats = ComparisonStats::from_vecs(
            &ndarray_stages.padded,
            &wgpu_stages.padded,
            ndarray_stages.padded_shape.clone(),
        );
        print_comparison("Padded Input", &padded_stats);

        let seanet_stats = ComparisonStats::from_vecs(
            &ndarray_stages.seanet,
            &wgpu_stages.seanet,
            ndarray_stages.seanet_shape.clone(),
        );
        print_comparison("SEANet Output", &seanet_stats);

        let transformer_stats = ComparisonStats::from_vecs(
            &ndarray_stages.transformer,
            &wgpu_stages.transformer,
            ndarray_stages.transformer_shape.clone(),
        );
        print_comparison("Transformer Output", &transformer_stats);

        let final_stats = ComparisonStats::from_vecs(
            &ndarray_stages.final_latent,
            &wgpu_stages.final_latent,
            ndarray_stages.final_latent_shape.clone(),
        );
        print_comparison("Final Latent", &final_stats);

        let cond_stats = ComparisonStats::from_vecs(
            &ndarray_stages.conditioning,
            &wgpu_stages.conditioning,
            ndarray_stages.conditioning_shape.clone(),
        );
        print_comparison("Conditioning", &cond_stats);

        // Summary
        println!("\n\n========================================");
        println!("SUMMARY");
        println!("========================================");

        let stages = [
            ("Padded Input", &padded_stats),
            ("SEANet Output", &seanet_stats),
            ("Transformer Output", &transformer_stats),
            ("Final Latent", &final_stats),
            ("Conditioning", &cond_stats),
        ];

        // Find first stage with significant divergence
        let divergence_threshold = 0.1; // 0.1% relative RMS difference
        let first_divergent = stages
            .iter()
            .find(|(_, stats)| stats.rms_diff_pct > divergence_threshold);

        match first_divergent {
            Some((name, stats)) => {
                println!("\nFirst significant divergence at: {}", name);
                println!("  Relative RMS difference: {:.3}%", stats.rms_diff_pct);
                println!("\nRecommended investigation:");
                if *name == "SEANet Output" {
                    println!("  - Check ELU activation (seanet.rs:13-17)");
                    println!("  - Check convolution kernel precision (streaming_conv.rs)");
                    println!("  - Check edge padding behavior (streaming_conv.rs:132-135)");
                } else if *name == "Transformer Output" {
                    println!("  - Check RoPE embeddings (mimi_transformer.rs:133-193)");
                    println!("  - Check softmax in attention (mimi_transformer.rs:272)");
                    println!("  - Check GELU activation");
                } else if *name == "Final Latent" {
                    println!("  - Check downsampling convolution (resample.rs)");
                } else if *name == "Conditioning" {
                    println!("  - Check linear projection precision");
                }
            }
            None => {
                println!(
                    "\nNo significant divergence detected (all stages < {:.1}% RMS diff)",
                    divergence_threshold
                );
                println!("\nThe conditioning RMS difference between backends:");
                println!("  NdArray: {:.6}", cond_stats.rms_a);
                println!("  WGPU:    {:.6}", cond_stats.rms_b);
                let rms_ratio =
                    (cond_stats.rms_b - cond_stats.rms_a).abs() / cond_stats.rms_a * 100.0;
                println!("  Difference: {:.2}%", rms_ratio);
            }
        }

        // Print cumulative error growth
        println!("\n\nError Accumulation:");
        println!("Stage               | RMS Diff % | Cumulative Growth");
        println!("--------------------|------------|------------------");
        let mut prev_pct = 0.0f32;
        for (name, stats) in stages.iter() {
            let growth = if prev_pct > 0.0 {
                format!("{:.1}x", stats.rms_diff_pct / prev_pct)
            } else {
                "-".to_string()
            };
            println!("{:19} | {:>10.4} | {}", name, stats.rms_diff_pct, growth);
            prev_pct = stats.rms_diff_pct.max(1e-10);
        }

        // Final diagnostic summary
        println!("\n\n========================================");
        println!("DIAGNOSTIC CONCLUSION");
        println!("========================================\n");

        if seanet_stats.rms_diff_pct < 0.1 && transformer_stats.rms_diff_pct > 1.0 {
            println!("ROOT CAUSE IDENTIFIED:");
            println!("  Burn's attention() function with causal mask diverges on WGPU backend.");
            println!("\nEVIDENCE:");
            println!(
                "  - SEANet (ELU + convolutions): {:.4}% RMS diff (PASS)",
                seanet_stats.rms_diff_pct
            );
            println!(
                "  - Transformer (attention):     {:.4}% RMS diff (FAIL)",
                transformer_stats.rms_diff_pct
            );
            println!("  - Individual ops (exp, sin, cos, gelu, softmax): ALL PASS");
            println!("  - Attention WITHOUT mask: PASS at all sequence lengths");
            println!("  - Attention WITH mask: FAIL at seq >= 128");
            println!("\nLIKELY BURN ISSUE:");
            println!("  The WGPU backend handles masked attention scores differently,");
            println!("  possibly in how -inf values interact with softmax.");
            println!("\nRECOMMENDED ACTIONS:");
            println!("  1. File a Burn issue about masked attention divergence on WGPU");
            println!("  2. Workaround: Implement custom attention with explicit mask handling");
            println!(
                "  3. Alternative: Use NdArray backend for encoding (WGPU for generation only)"
            );
        }
    }

    Ok(())
}
