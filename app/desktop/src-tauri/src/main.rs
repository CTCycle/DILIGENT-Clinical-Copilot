#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;
mod navigation;
mod runtime;
mod windows_job;

use std::sync::Mutex;
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder, WindowEvent};

struct BackendState(Mutex<Option<backend::BackendProcess>>);

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let version = app.package_info().version.to_string();
            let paths = runtime::prepare_runtime(&version).map_err(std::io::Error::other)?;
            let process =
                backend::BackendProcess::start(&paths, &version).map_err(std::io::Error::other)?;
            let base_url = process.base_url().to_owned();
            app.manage(BackendState(Mutex::new(Some(process))));
            let window =
                WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                    .title("DILIGENT Clinical Copilot")
                    .inner_size(1440.0, 960.0)
                    .visible(false)
                    .on_navigation(navigation::navigation_handler(base_url.clone()))
                    .build()
                    .map_err(std::io::Error::other)?;
            window
                .navigate(base_url.parse().map_err(std::io::Error::other)?)
                .map_err(std::io::Error::other)?;
            window.show().map_err(std::io::Error::other)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(event, WindowEvent::CloseRequested { .. }) {
                if let Some(state) = window.app_handle().try_state::<BackendState>() {
                    if let Ok(mut process) = state.inner().0.lock() {
                        if let Some(process) = process.as_mut() {
                            process.stop();
                        }
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running DILIGENT desktop");
}
