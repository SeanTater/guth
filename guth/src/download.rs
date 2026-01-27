use anyhow::Result;
use hf_hub::api::sync::Api;
use std::fs;
use std::path::{Path, PathBuf};

pub fn make_cache_directory() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").unwrap_or_else(|| ".".into());
    let cache_dir = Path::new(&home).join(".cache").join("guth");
    fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

pub fn download_if_necessary(path: &str) -> Result<PathBuf> {
    if let Some(stripped) = path.strip_prefix("hf://") {
        let (repo_id, filename, revision) = parse_hf_path(stripped)?;
        let api = Api::new()?;
        let repo = match revision {
            Some(rev) => api.repo(hf_hub::Repo::with_revision(repo_id, hf_hub::RepoType::Model, rev)),
            None => api.repo(hf_hub::Repo::model(repo_id)),
        };
        let cached = repo.get(&filename)?;
        return Ok(cached);
    }

    if path.starts_with("http://") || path.starts_with("https://") {
        anyhow::bail!("HTTP(S) download not implemented yet for {path}");
    }

    Ok(PathBuf::from(path))
}

fn parse_hf_path(path: &str) -> Result<(String, String, Option<String>)> {
    let mut parts = path.split('/').collect::<Vec<_>>();
    if parts.len() < 3 {
        anyhow::bail!("Invalid hf:// path: {path}");
    }

    let repo_id = format!("{}/{}", parts.remove(0), parts.remove(0));
    let filename = parts.join("/");

    if let Some((file, rev)) = filename.split_once('@') {
        return Ok((repo_id, file.to_string(), Some(rev.to_string())));
    }

    Ok((repo_id, filename, None))
}
