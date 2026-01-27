use anyhow::Result;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

#[derive(Debug, Default)]
pub struct WavIo;

impl WavIo {
    pub fn read_wav(path: impl AsRef<Path>) -> Result<(Vec<Vec<f32>>, u32)> {
        let mut reader = WavReader::open(path)?;
        let spec = reader.spec();
        let channels = spec.channels as usize;
        let sample_rate = spec.sample_rate;
        let mut samples = vec![Vec::new(); channels];

        match spec.sample_format {
            SampleFormat::Float => {
                for (idx, sample) in reader.samples::<f32>().enumerate() {
                    let value = sample?;
                    samples[idx % channels].push(value);
                }
            }
            SampleFormat::Int => {
                let max = (1_i64 << (spec.bits_per_sample - 1)) as f32;
                for (idx, sample) in reader.samples::<i32>().enumerate() {
                    let value = sample? as f32 / max;
                    samples[idx % channels].push(value);
                }
            }
        }

        Ok((samples, sample_rate))
    }

    pub fn write_wav(
        path: impl AsRef<Path>,
        samples: &[Vec<f32>],
        sample_rate: u32,
    ) -> Result<()> {
        if samples.is_empty() {
            anyhow::bail!("No audio channels provided");
        }
        let channels = samples.len() as u16;
        let len = samples[0].len();
        for channel in samples.iter().skip(1) {
            if channel.len() != len {
                anyhow::bail!("Channel length mismatch in WAV write");
            }
        }

        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec)?;

        for idx in 0..len {
            for channel in samples {
                let value = channel[idx].clamp(-1.0, 1.0);
                let scaled = (value * i16::MAX as f32).round() as i16;
                writer.write_sample(scaled)?;
            }
        }

        writer.finalize()?;
        Ok(())
    }
}

pub struct StreamingWavWriter {
    writer: WavWriter<BufWriter<File>>,
    channels: usize,
}

impl StreamingWavWriter {
    pub fn create(path: impl AsRef<Path>, sample_rate: u32, channels: usize) -> Result<Self> {
        let spec = WavSpec {
            channels: channels as u16,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let writer = WavWriter::create(path, spec)?;
        Ok(Self { writer, channels })
    }

    pub fn write_chunk(&mut self, samples: &[Vec<f32>]) -> Result<()> {
        if samples.len() != self.channels {
            anyhow::bail!("Streaming WAV chunk has wrong channel count");
        }
        let len = samples[0].len();
        for channel in samples.iter().skip(1) {
            if channel.len() != len {
                anyhow::bail!("Streaming WAV chunk length mismatch");
            }
        }
        for idx in 0..len {
            for channel in samples {
                let value = channel[idx].clamp(-1.0, 1.0);
                let scaled = (value * i16::MAX as f32).round() as i16;
                self.writer.write_sample(scaled)?;
            }
        }
        Ok(())
    }

    pub fn finalize(self) -> Result<()> {
        self.writer.finalize()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::WavIo;
    use tempfile::tempdir;

    #[test]
    fn wav_roundtrip_preserves_shape() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("test.wav");
        let samples = vec![vec![0.0_f32, 0.5, -0.25], vec![0.1, -0.1, 0.2]];
        WavIo::write_wav(&path, &samples, 24000).expect("write wav");

        let (decoded, sample_rate) = WavIo::read_wav(&path).expect("read wav");
        assert_eq!(sample_rate, 24000);
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].len(), 3);
    }
}
