use anyhow::Result;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

#[derive(Debug, Default)]
pub struct AudioResampler;

impl AudioResampler {
    pub fn convert_audio(
        mut samples: Vec<Vec<f32>>,
        from_rate: u32,
        to_rate: u32,
        to_channels: usize,
    ) -> Result<Vec<Vec<f32>>> {
        samples = convert_channels(samples, to_channels)?;
        if from_rate == to_rate || samples.is_empty() || samples[0].is_empty() {
            return Ok(samples);
        }

        let channels = samples.len();
        let input_len = samples[0].len();
        let ratio = to_rate as f64 / from_rate as f64;
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, input_len, channels)?;
        let output = resampler.process(&samples, None)?;
        Ok(output)
    }
}

fn convert_channels(samples: Vec<Vec<f32>>, to_channels: usize) -> Result<Vec<Vec<f32>>> {
    if samples.is_empty() {
        return Ok(samples);
    }
    let from_channels = samples.len();
    if from_channels == to_channels {
        return Ok(samples);
    }
    if to_channels == 1 {
        let len = samples[0].len();
        let mut mixed = vec![0.0_f32; len];
        for channel in &samples {
            if channel.len() != len {
                anyhow::bail!("Channel length mismatch in audio conversion");
            }
            for (idx, value) in channel.iter().enumerate() {
                mixed[idx] += *value;
            }
        }
        let scale = 1.0 / from_channels as f32;
        for value in &mut mixed {
            *value *= scale;
        }
        return Ok(vec![mixed]);
    }
    if from_channels == 1 && to_channels > 1 {
        let mut duplicated = Vec::with_capacity(to_channels);
        for _ in 0..to_channels {
            duplicated.push(samples[0].clone());
        }
        return Ok(duplicated);
    }
    anyhow::bail!(
        "Unsupported channel conversion from {from_channels} to {to_channels}"
    )
}

#[cfg(test)]
mod tests {
    use super::AudioResampler;

    #[test]
    fn converts_channels_and_resamples() {
        let samples = vec![vec![0.0_f32; 480]];
        let output = AudioResampler::convert_audio(samples, 48000, 24000, 2)
            .expect("convert audio");
        assert_eq!(output.len(), 2);
        assert!(!output[0].is_empty());
    }
}
