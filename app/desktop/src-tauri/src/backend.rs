use std::{
    fs::{self, OpenOptions},
    path::PathBuf,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use reqwest::blocking::Client;
use serde::Deserialize;

use crate::{runtime::RuntimePaths, windows_job::WindowsJob};

#[derive(Debug, Deserialize)]
struct ReadyFile {
    port: u16,
    pid: u32,
    release_version: String,
}

pub struct BackendProcess {
    child: Child,
    _job: WindowsJob,
    ready_file: PathBuf,
    base_url: String,
}

fn rotate_log(path: &PathBuf) -> Result<(), String> {
    if path.is_file()
        && fs::metadata(path).map_err(|error| error.to_string())?.len() > 5 * 1024 * 1024
    {
        let previous = path.with_extension("log.1");
        let _ = fs::remove_file(&previous);
        fs::rename(path, previous).map_err(|error| error.to_string())?;
    }
    Ok(())
}

impl BackendProcess {
    pub fn start(paths: &RuntimePaths, version: &str) -> Result<Self, String> {
        let state_root = paths.data_root.join("state");
        let log_root = paths.data_root.join("resources").join("logs");
        fs::create_dir_all(&state_root).map_err(|error| error.to_string())?;
        fs::create_dir_all(&log_root).map_err(|error| error.to_string())?;
        let ready_file = state_root.join("desktop-backend-ready.json");
        let log_path = log_root.join("desktop-backend.log");
        rotate_log(&log_path)?;
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|error| error.to_string())?;
        let log_stderr = log.try_clone().map_err(|error| error.to_string())?;
        let mut command = Command::new(&paths.backend);
        command
            .args([
                "--ready-file",
                ready_file.to_string_lossy().as_ref(),
                "--host",
                "127.0.0.1",
            ])
            .current_dir(&paths.root)
            .env("DILIGENT_DESKTOP", "true")
            .env("DILIGENT_RELEASE_VERSION", version)
            .env("DILIGENT_RUNTIME_ROOT", &paths.root)
            .env("DILIGENT_DATA_ROOT", &paths.data_root)
            .env(
                "DILIGENT_SQLITE_PATH",
                paths.data_root.join("resources").join("database.db"),
            )
            .env(
                "DILIGENT_ACCESS_KEY_MATERIAL_FILE",
                paths
                    .data_root
                    .join("resources")
                    .join("access-key-material.json"),
            )
            .env("RELOAD", "false")
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(log_stderr));
        #[cfg(windows)]
        std::os::windows::process::CommandExt::creation_flags(&mut command, 0x08000000);
        let mut child = command
            .spawn()
            .map_err(|error| format!("unable to start backend: {error}"))?;
        let job = WindowsJob::attach(&child)?;
        let ready = wait_for_ready(&mut child, &ready_file, version)?;
        let base_url = format!("http://127.0.0.1:{}", ready.port);
        let health_url = format!("{base_url}/api/health");
        let client = Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|error| error.to_string())?;
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            if child
                .try_wait()
                .map_err(|error| error.to_string())?
                .is_some()
            {
                return Err("packaged backend exited before health check".into());
            }
            if client
                .get(&health_url)
                .send()
                .is_ok_and(|response| response.status().is_success())
            {
                break;
            }
            if Instant::now() >= deadline {
                return Err(format!("backend health timeout: {health_url}"));
            }
            thread::sleep(Duration::from_millis(250));
        }
        Ok(Self {
            child,
            _job: job,
            ready_file,
            base_url,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = fs::remove_file(&self.ready_file);
    }
}

impl Drop for BackendProcess {
    fn drop(&mut self) {
        self.stop();
    }
}

fn wait_for_ready(child: &mut Child, path: &PathBuf, version: &str) -> Result<ReadyFile, String> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if path.is_file() {
            let payload = fs::read(path).map_err(|error| error.to_string())?;
            let ready: ReadyFile =
                serde_json::from_slice(&payload).map_err(|error| error.to_string())?;
            if ready.port == 0 || ready.pid != child.id() || ready.release_version != version {
                return Err("invalid backend ready-file contract".into());
            }
            return Ok(ready);
        }
        if child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_some()
        {
            return Err("packaged backend exited before ready file".into());
        }
        if Instant::now() >= deadline {
            return Err("backend ready-file timeout".into());
        }
        thread::sleep(Duration::from_millis(100));
    }
}
