// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

use tauri::Url;

pub fn navigation_handler() -> impl Fn(&Url) -> bool + Send + 'static {
    move |url| {
        let target = url.as_str();
        let local_app = (url.scheme() == "tauri" && url.host_str() == Some("localhost"))
            || (matches!(url.scheme(), "http" | "https")
                && url.host_str() == Some("tauri.localhost")
                && url.port().is_none());
        if local_app {
            true
        } else if matches!(url.scheme(), "http" | "https" | "mailto") {
            let _ = open::that(target);
            false
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::navigation_handler;
    use tauri::Url;

    #[test]
    fn allows_tauri_app_routes_and_rejects_file_urls() {
        let handler = navigation_handler();

        assert!(handler(&Url::parse("http://tauri.localhost/").unwrap()));
        assert!(handler(&Url::parse("http://tauri.localhost/clinical-sessions").unwrap()));
        assert!(!handler(&Url::parse("file:///C:/outside.html").unwrap()));
    }
}
