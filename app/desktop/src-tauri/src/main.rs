#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;
mod navigation;
mod runtime;
mod windows_job;

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder, WindowEvent};
use uuid::Uuid;

struct BackendState(Mutex<Option<backend::BackendProcess>>);
struct StartupState {
    cancelled: AtomicBool,
}

fn show_startup_error(window: &tauri::WebviewWindow, message: &str) {
    let Ok(payload) = serde_json::to_string(message) else {
        return;
    };
    let _ = window.eval(format!("window.showStartupError({payload});"));
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }))
        .manage(BackendState(Mutex::new(None)))
        .manage(StartupState {
            cancelled: AtomicBool::new(false),
        })
        .setup(|app| {
            let version = app.package_info().version.to_string();
            let backend_origin = Arc::new(Mutex::new(None));
            let window =
                WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                    .title("DILIGENT Clinical Copilot")
                    .inner_size(1440.0, 960.0)
                    .visible(true)
                    .on_navigation(navigation::navigation_handler(backend_origin.clone()))
                    .build()
                    .map_err(std::io::Error::other)?;
            window.show().map_err(std::io::Error::other)?;

            let app_handle = app.handle().clone();
            let startup_window = window.clone();
            std::thread::spawn(move || {
                let session_secret = Uuid::new_v4().to_string();
                let startup_result = runtime::prepare_runtime(&version).and_then(|paths| {
                    runtime::prune_runtime_cache(&version);
                    let process = backend::BackendProcess::start(&paths, &version, &session_secret)?;
                    Ok((paths, process))
                });
                match startup_result {
                    Ok((paths, mut process)) => {
                        let startup = app_handle.state::<StartupState>();
                        if startup.cancelled.load(Ordering::Acquire) {
                            process.stop();
                            return;
                        }
                        let base_url = process.base_url().to_owned();
                        if let Ok(mut origin) = backend_origin.lock() {
                            *origin = Some(base_url.clone());
                        }
                        if let Ok(mut state) = app_handle.state::<BackendState>().0.lock() {
                            *state = Some(process);
                        }
                        let bootstrap_url = format!(
                            "{base_url}/#desktop-bootstrap={session_secret}"
                        );
                        let navigation_result = bootstrap_url
                            .parse()
                            .map_err(std::io::Error::other)
                            .and_then(|url| {
                                startup_window
                                    .navigate(url)
                                    .map_err(std::io::Error::other)
                            });
                        if navigation_result.is_err() {
                            if let Ok(mut state) = app_handle.state::<BackendState>().0.lock() {
                                if let Some(process) = state.as_mut() {
                                    process.stop();
                                }
                                *state = None;
                            }
                            show_startup_error(
                                &startup_window,
                                "DILIGENT could not open its local interface. Please restart the application.",
                            );
                            return;
                        }
                        let _ = paths;
                    }
                    Err(_) => show_startup_error(
                        &startup_window,
                        "DILIGENT could not start its local services. Please restart the application.",
                    ),
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(event, WindowEvent::CloseRequested { .. }) {
                if let Some(startup) = window.app_handle().try_state::<StartupState>() {
                    startup.cancelled.store(true, Ordering::Release);
                }
                if let Some(state) = window.app_handle().try_state::<BackendState>() {
                    let process = state.inner().0.lock().ok().and_then(|mut value| value.take());
                    if let Some(mut process) = process {
                        process.stop();
                    }
                }
                window.app_handle().exit(0);
                std::process::exit(0);
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running DILIGENT desktop");
}
