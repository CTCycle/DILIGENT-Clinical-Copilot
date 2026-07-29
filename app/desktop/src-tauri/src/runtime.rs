use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use uuid::Uuid;
use zip::ZipArchive;

const EMBEDDED_ARCHIVE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/generated/diligent-runtime.zip"
));
const EMBEDDED_DIGEST: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/generated/diligent-runtime.sha256"
));

#[derive(Debug, Clone)]
pub struct RuntimePaths {
    pub root: PathBuf,
    pub data_root: PathBuf,
    pub backend: PathBuf,
}

#[derive(Debug, Deserialize)]
struct RuntimeManifest {
    release_version: String,
    files: Vec<RuntimeFile>,
}

#[derive(Debug, Deserialize)]
struct RuntimeFile {
    path: String,
    size: u64,
    sha256: String,
}

fn archive_digest() -> String {
    format!("{:x}", Sha256::digest(EMBEDDED_ARCHIVE))
}

fn safe_member_path(name: &str) -> Result<PathBuf, String> {
    let path = Path::new(name);
    if path.is_absolute()
        || name.starts_with('/')
        || name.starts_with('\\')
        || name.contains(':')
        || path
            .components()
            .any(|component| component == std::path::Component::ParentDir)
    {
        return Err(format!("unsafe runtime archive member: {name}"));
    }
    Ok(path.to_path_buf())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_extracted_runtime(root: &Path, expected_version: &str) -> Result<(), String> {
    let manifest_path = root.join("runtime-manifest.json");
    let manifest: RuntimeManifest =
        serde_json::from_slice(&fs::read(&manifest_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if manifest.release_version != expected_version {
        return Err("runtime version mismatch".into());
    }
    for file in manifest.files {
        let relative = safe_member_path(&file.path)?;
        let path = root.join(relative);
        let metadata = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
        if !metadata.is_file() || metadata.len() != file.size || sha256_file(&path)? != file.sha256
        {
            return Err(format!("runtime manifest mismatch: {}", file.path));
        }
    }
    if !root.join("backend").join("DILIGENTBackend.exe").is_file() {
        return Err("packaged backend executable is missing".into());
    }
    Ok(())
}

fn local_app_root() -> Result<PathBuf, String> {
    std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .ok_or_else(|| "LOCALAPPDATA is not available".into())
}

pub fn prepare_runtime(version: &str) -> Result<RuntimePaths, String> {
    let actual_digest = archive_digest();
    if actual_digest != EMBEDDED_DIGEST.trim() {
        return Err("embedded runtime digest mismatch".into());
    }
    let app_root = local_app_root()?.join("DILIGENT");
    let runtime_parent = app_root.join("runtime").join(version);
    let runtime_root = runtime_parent.join(&actual_digest);
    if !runtime_root.join("extraction.complete").is_file()
        || validate_extracted_runtime(&runtime_root, version).is_err()
    {
        fs::create_dir_all(&runtime_parent).map_err(|error| error.to_string())?;
        let temporary = runtime_parent.join(format!(".extract-{}", Uuid::new_v4()));
        fs::create_dir_all(&temporary).map_err(|error| error.to_string())?;
        let extraction_result = (|| {
            let cursor = std::io::Cursor::new(EMBEDDED_ARCHIVE);
            let mut archive = ZipArchive::new(cursor).map_err(|error| error.to_string())?;
            for index in 0..archive.len() {
                let mut member = archive.by_index(index).map_err(|error| error.to_string())?;
                let relative = safe_member_path(member.name())?;
                if member.is_dir() {
                    fs::create_dir_all(temporary.join(relative))
                        .map_err(|error| error.to_string())?;
                    continue;
                }
                if member
                    .unix_mode()
                    .is_some_and(|mode| mode & 0o170000 == 0o120000)
                {
                    return Err("symlink in runtime archive".into());
                }
                let destination = temporary.join(relative);
                if let Some(parent) = destination.parent() {
                    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
                }
                let mut output =
                    fs::File::create(destination).map_err(|error| error.to_string())?;
                std::io::copy(&mut member, &mut output).map_err(|error| error.to_string())?;
            }
            validate_extracted_runtime(&temporary, version)?;
            fs::write(temporary.join("extraction.complete"), b"complete\n")
                .map_err(|error| error.to_string())?;
            if runtime_root.exists() {
                fs::remove_dir_all(&runtime_root).map_err(|error| error.to_string())?;
            }
            fs::rename(&temporary, &runtime_root).map_err(|error| error.to_string())
        })();
        if extraction_result.is_err() {
            let _ = fs::remove_dir_all(&temporary);
        }
        extraction_result?;
    }
    let data_root = app_root.join("data");
    fs::create_dir_all(data_root.join("settings")).map_err(|error| error.to_string())?;
    fs::create_dir_all(data_root.join("resources").join("logs"))
        .map_err(|error| error.to_string())?;
    Ok(RuntimePaths {
        backend: runtime_root.join("backend").join("DILIGENTBackend.exe"),
        root: runtime_root,
        data_root,
    })
}

#[cfg(test)]
mod tests {
    use super::safe_member_path;

    #[test]
    fn rejects_traversal_and_absolute_members() {
        assert!(safe_member_path("../escape").is_err());
        assert!(safe_member_path("C:/escape").is_err());
        assert!(safe_member_path("/escape").is_err());
    }
}
