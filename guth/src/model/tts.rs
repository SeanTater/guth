use anyhow::Result;

#[derive(Debug, Default)]
pub struct TtsModel;

impl TtsModel {
    pub fn load() -> Result<Self> {
        Ok(Self)
    }
}
