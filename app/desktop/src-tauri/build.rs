use std::{env, fs, path::PathBuf};

use sha2::{Digest, Sha256};

fn main() {
    tauri_build::build();
    let generated = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("generated");
    fs::create_dir_all(&generated).unwrap();
    let archive = generated.join("diligent-runtime.zip");
    if !archive.exists() {
        // An empty archive keeps cargo check/test useful before a release build.
        fs::write(
            &archive,
            [
                0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        )
        .unwrap();
    }
    let digest = Sha256::digest(fs::read(&archive).unwrap());
    fs::write(
        generated.join("diligent-runtime.sha256"),
        format!("{digest:x}\n"),
    )
    .unwrap();
    println!("cargo:rerun-if-changed={}", archive.display());
}
