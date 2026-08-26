use std::{
    fs::{self, OpenOptions},
    io::Read,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;
use zip::ZipArchive;

const EMBEDDED_ARCHIVE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/diligent-runtime.zip"));
const EMBEDDED_DIGEST: &str = include_str!(concat!(env!("OUT_DIR"), "/diligent-runtime.sha256"));

#[derive(Debug, Clone)]
pub struct RuntimePaths {
    pub root: PathBuf,
    pub data_root: PathBuf,
    pub backend: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
struct RuntimeManifest {
    release_version: String,
    files: Vec<RuntimeFile>,
}

#[derive(Debug, Clone, Deserialize)]
struct RuntimeFile {
    path: String,
    size: u64,
    sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExtractionMarker {
    release_version: String,
    archive_digest: String,
    manifest_sha256: String,
    file_count: usize,
    total_size: u64,
}

struct RuntimeLock(PathBuf);

impl Drop for RuntimeLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn archive_digest() -> String {
    format!("{:x}", Sha256::digest(EMBEDDED_ARCHIVE))
}

fn safe_member_path(name: &str) -> Result<PathBuf, String> {
    let path = Path::new(name);
    if name.trim().is_empty()
        || path.is_absolute()
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
    // Keep the hashing buffer on the heap. The Windows GUI subsystem uses a
    // relatively small default thread stack, so a 1 MiB stack array can
    // overflow before the desktop window is created.
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn load_manifest(root: &Path, expected_version: &str) -> Result<(RuntimeManifest, String), String> {
    let manifest_bytes =
        fs::read(root.join("runtime-manifest.json")).map_err(|error| error.to_string())?;
    let manifest: RuntimeManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|error| error.to_string())?;
    if manifest.release_version != expected_version {
        return Err("runtime version mismatch".into());
    }
    Ok((manifest, format!("{:x}", Sha256::digest(manifest_bytes))))
}

fn validate_manifest_entries(
    root: &Path,
    manifest: &RuntimeManifest,
    hash_files: bool,
) -> Result<(usize, u64), String> {
    let mut total_size = 0_u64;
    for file in &manifest.files {
        let relative = safe_member_path(&file.path)?;
        let path = root.join(relative);
        let metadata = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
        if !metadata.is_file() || metadata.len() != file.size {
            return Err(format!("runtime manifest size mismatch: {}", file.path));
        }
        if hash_files && sha256_file(&path)? != file.sha256 {
            return Err(format!("runtime manifest hash mismatch: {}", file.path));
        }
        total_size = total_size
            .checked_add(file.size)
            .ok_or_else(|| "runtime manifest size overflow".to_string())?;
    }
    if !root.join("backend").join("DILIGENTBackend.exe").is_file() {
        return Err("packaged backend executable is missing".into());
    }
    Ok((manifest.files.len(), total_size))
}

fn read_marker(path: &Path) -> Result<ExtractionMarker, String> {
    serde_json::from_slice(&fs::read(path).map_err(|error| error.to_string())?)
        .map_err(|error| error.to_string())
}

fn fast_validate_extracted_runtime(
    root: &Path,
    expected_version: &str,
    expected_digest: &str,
) -> Result<(), String> {
    let marker = read_marker(&root.join("extraction.complete"))?;
    if marker.release_version != expected_version || marker.archive_digest != expected_digest {
        return Err("runtime extraction marker mismatch".into());
    }
    let (manifest, manifest_sha256) = load_manifest(root, expected_version)?;
    if marker.manifest_sha256 != manifest_sha256 {
        return Err("runtime manifest marker mismatch".into());
    }
    let (file_count, total_size) = validate_manifest_entries(root, &manifest, false)?;
    if marker.file_count != file_count || marker.total_size != total_size {
        return Err("runtime extraction statistics mismatch".into());
    }
    Ok(())
}

fn acquire_runtime_lock(path: &Path) -> Result<RuntimeLock, String> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(file) => {
                drop(file);
                return Ok(RuntimeLock(path.to_path_buf()));
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if Instant::now() >= deadline {
                    return Err("timed out waiting for runtime extraction lock".into());
                }
                thread::sleep(Duration::from_millis(100));
            }
            Err(error) => return Err(error.to_string()),
        }
    }
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
    let runtime_base = app_root.join("runtime");
    let runtime_parent = runtime_base.join(version);
    let runtime_root = runtime_parent.join(&actual_digest);
    fs::create_dir_all(&runtime_base).map_err(|error| error.to_string())?;
    let _lock = acquire_runtime_lock(&runtime_base.join(".runtime.lock"))?;

    if fast_validate_extracted_runtime(&runtime_root, version, &actual_digest).is_err() {
        fs::create_dir_all(&runtime_parent).map_err(|error| error.to_string())?;
        let temporary = runtime_parent.join(format!(".extract-{}", Uuid::new_v4()));
        fs::create_dir_all(&temporary).map_err(|error| error.to_string())?;
        let extraction_result = (|| {
            let cursor = std::io::Cursor::new(EMBEDDED_ARCHIVE);
            let mut archive = ZipArchive::new(cursor).map_err(|error| error.to_string())?;
            let mut members = std::collections::HashSet::new();
            for index in 0..archive.len() {
                let mut member = archive.by_index(index).map_err(|error| error.to_string())?;
                let relative = safe_member_path(member.name())?;
                if !members.insert(relative.clone()) {
                    return Err(format!(
                        "duplicate runtime archive member: {}",
                        member.name()
                    ));
                }
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
            let (manifest, manifest_sha256) = load_manifest(&temporary, version)?;
            let (file_count, total_size) = validate_manifest_entries(&temporary, &manifest, true)?;
            let marker = ExtractionMarker {
                release_version: version.to_string(),
                archive_digest: actual_digest.clone(),
                manifest_sha256,
                file_count,
                total_size,
            };
            fs::write(
                temporary.join("extraction.complete"),
                serde_json::to_vec_pretty(&marker).map_err(|error| error.to_string())?,
            )
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

pub fn prune_runtime_cache(version: &str) {
    let Ok(app_root) = local_app_root().map(|path| path.join("DILIGENT")) else {
        return;
    };
    let runtime_base = app_root.join("runtime");
    let Ok(entries) = fs::read_dir(&runtime_base) else {
        return;
    };
    for (inspected, entry) in entries.flatten().enumerate() {
        if inspected >= 64 {
            break;
        }
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == ".runtime.lock" {
            continue;
        }
        if name != version {
            if path.is_dir() {
                let _ = fs::remove_dir_all(path);
            }
            continue;
        }
        let Ok(children) = fs::read_dir(path) else {
            continue;
        };
        for (child_count, child) in children.flatten().enumerate() {
            if child_count >= 64 {
                break;
            }
            let child_path = child.path();
            let child_name = child.file_name().to_string_lossy().into_owned();
            if child_name.starts_with(".extract-") {
                let _ = fs::remove_dir_all(child_path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::safe_member_path;

    #[test]
    fn rejects_traversal_and_absolute_members() {
        assert!(safe_member_path("../escape").is_err());
        assert!(safe_member_path("C:/escape").is_err());
        assert!(safe_member_path("/escape").is_err());
        assert!(safe_member_path("\\escape").is_err());
    }
}
