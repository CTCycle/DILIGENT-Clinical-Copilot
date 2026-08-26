use std::{env, fs, path::PathBuf};

use sha2::{Digest, Sha256};

fn main() {
    tauri_build::build();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::create_dir_all(&out_dir).unwrap();
    let archive = out_dir.join("diligent-runtime.zip");
    let configured_archive = env::var_os("DILIGENT_RUNTIME_ARCHIVE").map(PathBuf::from);
    let source_archive = configured_archive
        .as_ref()
        .filter(|path| path.is_file())
        .cloned();
    if let Some(source_archive) = source_archive {
        fs::copy(&source_archive, &archive).unwrap();
        println!("cargo:rerun-if-changed={}", source_archive.display());
    } else if env::var("PROFILE").as_deref() == Ok("debug") {
        // An empty archive keeps cargo check/test useful before a release build.
        fs::write(
            &archive,
            [
                0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        )
        .unwrap();
    } else {
        panic!("DILIGENT_RUNTIME_ARCHIVE must point to a built runtime ZIP for release builds");
    }
    println!("cargo:rerun-if-env-changed=DILIGENT_RUNTIME_ARCHIVE");
    let digest = Sha256::digest(fs::read(&archive).unwrap());
    fs::write(
        out_dir.join("diligent-runtime.sha256"),
        format!("{digest:x}\n"),
    )
    .unwrap();
}
