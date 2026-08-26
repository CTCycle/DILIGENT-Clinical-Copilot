use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    net::TcpStream,
    path::PathBuf,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

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
    session_secret: String,
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

fn terminate_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

impl BackendProcess {
    pub fn start(
        paths: &RuntimePaths,
        version: &str,
        session_secret: &str,
    ) -> Result<Self, String> {
        let state_root = paths.data_root.join("state");
        let log_root = paths.data_root.join("resources").join("logs");
        fs::create_dir_all(&state_root).map_err(|error| error.to_string())?;
        fs::create_dir_all(&log_root).map_err(|error| error.to_string())?;
        let ready_file = state_root.join("desktop-backend-ready.json");
        if ready_file.is_file() {
            fs::remove_file(&ready_file).map_err(|error| error.to_string())?;
        }
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
            .env("DILIGENT_DESKTOP_SESSION_SECRET", session_secret)
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
        let job = match WindowsJob::attach(&child) {
            Ok(job) => job,
            Err(error) => {
                terminate_child(&mut child);
                return Err(error);
            }
        };
        let ready = match wait_for_ready(&mut child, &ready_file, version) {
            Ok(ready) => ready,
            Err(error) => {
                terminate_child(&mut child);
                return Err(error);
            }
        };
        let base_url = format!("http://127.0.0.1:{}", ready.port);
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            match child.try_wait() {
                Ok(Some(_)) => return Err("packaged backend exited before health check".into()),
                Err(error) => {
                    terminate_child(&mut child);
                    return Err(error.to_string());
                }
                Ok(None) => {}
            }
            if loopback_request(&base_url, "GET", "/api/health", None, None)
                .is_ok_and(|status| (200..300).contains(&status))
            {
                break;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("backend health timeout: {base_url}/api/health"));
            }
            thread::sleep(Duration::from_millis(250));
        }
        Ok(Self {
            child,
            _job: job,
            ready_file,
            base_url,
            session_secret: session_secret.to_string(),
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn stop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let cookie = format!("diligent_desktop_session={}", self.session_secret);
            let _ = loopback_request(
                &self.base_url,
                "POST",
                "/api/desktop/shutdown",
                Some("{}"),
                Some(&cookie),
            );
            let deadline = Instant::now() + Duration::from_secs(8);
            loop {
                match self.child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) if Instant::now() < deadline => {
                        thread::sleep(Duration::from_millis(100));
                    }
                    _ => {
                        terminate_child(&mut self.child);
                        break;
                    }
                }
            }
        }
        let _ = fs::remove_file(&self.ready_file);
    }
}

impl Drop for BackendProcess {
    fn drop(&mut self) {
        self.stop();
    }
}

fn loopback_request(
    base_url: &str,
    method: &str,
    path: &str,
    body: Option<&str>,
    cookie: Option<&str>,
) -> Result<u16, String> {
    let port = base_url
        .strip_prefix("http://127.0.0.1:")
        .ok_or_else(|| "backend origin is not loopback HTTP".to_string())?
        .parse::<u16>()
        .map_err(|error| error.to_string())?;
    let address = format!("127.0.0.1:{port}");
    let mut stream = TcpStream::connect_timeout(
        &address
            .parse()
            .map_err(|error: std::net::AddrParseError| error.to_string())?,
        Duration::from_millis(500),
    )
    .map_err(|error| error.to_string())?;
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|error| error.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .map_err(|error| error.to_string())?;

    let payload = body.unwrap_or("");
    let mut request =
        format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n");
    if method != "GET" {
        request.push_str(&format!("Origin: {base_url}\r\n"));
    }
    if let Some(cookie) = cookie {
        request.push_str(&format!("Cookie: {cookie}\r\n"));
    }
    if !payload.is_empty() {
        request.push_str(&format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            payload.len()
        ));
    }
    request.push_str("\r\n");
    request.push_str(payload);
    stream
        .write_all(request.as_bytes())
        .map_err(|error| error.to_string())?;

    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|error| error.to_string())?;
    let status_line = response
        .split(|byte| *byte == b'\n')
        .next()
        .ok_or_else(|| "backend returned an empty HTTP response".to_string())?;
    let status_bytes = status_line
        .split(|byte| *byte == b' ')
        .nth(1)
        .ok_or_else(|| "backend returned an invalid HTTP status line".to_string())?;
    let status = std::str::from_utf8(status_bytes)
        .map_err(|error| error.to_string())?
        .trim()
        .parse::<u16>()
        .map_err(|error| error.to_string())?;
    Ok(status)
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
