//! Utilities for downloading model weights from HuggingFace Hub and HTTP URLs.
//!
//! Files are cached in `~/.cache/guth/` to avoid re-downloading.

use anyhow::Result;
use hf_hub::api::sync::Api;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Create the cache directory if it doesn't exist.
///
/// Returns the path to `~/.cache/guth/`.
pub fn make_cache_directory() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").unwrap_or_else(|| ".".into());
    let cache_dir = Path::new(&home).join(".cache").join("guth");
    fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

/// Download a file if it's a remote URL, or return the local path unchanged.
///
/// Supports three path formats:
/// - `hf://owner/repo/path/to/file.bin` - HuggingFace Hub
/// - `hf://owner/repo/path/to/file.bin@revision` - HuggingFace with specific revision
/// - `https://example.com/file.bin` - Direct HTTP download
/// - `/local/path/file.bin` - Local file (returned as-is)
///
/// Downloaded files are cached in `~/.cache/guth/`.
pub fn download_if_necessary(path: &str) -> Result<PathBuf> {
    if let Some(stripped) = path.strip_prefix("hf://") {
        let (repo_id, filename, revision) = parse_hf_path(stripped)?;
        let api = Api::new()?;
        let repo = match revision {
            Some(rev) => api.repo(hf_hub::Repo::with_revision(
                repo_id,
                hf_hub::RepoType::Model,
                rev,
            )),
            None => api.repo(hf_hub::Repo::model(repo_id)),
        };
        let cached = repo.get(&filename)?;
        return Ok(cached);
    }

    if path.starts_with("http://") || path.starts_with("https://") {
        return download_http(path);
    }

    let local = PathBuf::from(path);
    if !local.exists() {
        anyhow::bail!("No such file or directory: {path}");
    }
    Ok(local)
}

/// Download a file from an HTTP(S) URL to the cache directory.
fn download_http(url: &str) -> Result<PathBuf> {
    let cache_dir = make_cache_directory()?;

    // Create a deterministic filename from the URL
    let filename = url_to_cache_filename(url);
    let cache_path = cache_dir.join(&filename);

    // If already cached, return the path
    if cache_path.exists() {
        return Ok(cache_path);
    }

    eprintln!("Downloading {url}...");

    let response = ureq::get(url)
        .call()
        .map_err(|e| anyhow::anyhow!("Failed to download {url}: {e}"))?;

    // Read the response body into memory then write to file
    let mut data = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut data)
        .map_err(|e| anyhow::anyhow!("Failed to read response: {e}"))?;

    // Write to a temporary file first, then rename (atomic on most filesystems)
    let temp_path = cache_path.with_extension("tmp");
    let mut file = fs::File::create(&temp_path)?;
    file.write_all(&data)?;
    file.sync_all()?;
    drop(file);

    fs::rename(&temp_path, &cache_path)?;

    eprintln!("Downloaded to {}", cache_path.display());
    Ok(cache_path)
}

/// Convert a URL to a safe cache filename.
fn url_to_cache_filename(url: &str) -> String {
    // Strip protocol and replace unsafe characters
    let stripped = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);

    // Replace path separators and other special characters
    stripped
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '?' | '&' | '=' | '#' => '_',
            c if c.is_ascii_alphanumeric() || c == '.' || c == '-' => c,
            _ => '_',
        })
        .collect()
}

/// Parse `owner/repo/path@rev` into components for HuggingFace downloads.
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

#[cfg(test)]
mod tests {
    use super::{download_if_necessary, url_to_cache_filename};

    #[test]
    fn download_rejects_invalid_hf_path() {
        let err = download_if_necessary("hf://too-short").unwrap_err();
        assert!(err.to_string().contains("Invalid hf:// path"));
    }

    #[test]
    fn download_accepts_existing_local_path() {
        let path = download_if_necessary("tests/fixtures/tokenizer.model").unwrap();
        assert!(path.ends_with("tests/fixtures/tokenizer.model"));
    }

    #[test]
    fn download_rejects_missing_local_path() {
        let err = download_if_necessary("tests/fixtures/missing_local_file.bin").unwrap_err();
        assert!(err.to_string().to_lowercase().contains("no such file"));
    }

    #[test]
    fn url_to_cache_filename_handles_special_chars() {
        assert_eq!(
            url_to_cache_filename("https://example.com/path/to/file.bin"),
            "example.com_path_to_file.bin"
        );
        assert_eq!(
            url_to_cache_filename("https://example.com/file?query=1&other=2"),
            "example.com_file_query_1_other_2"
        );
    }
}
