use tauri::Url;

pub fn navigation_handler(backend_origin: String) -> impl Fn(&Url) -> bool + Send + 'static {
    move |url| {
        let target = url.as_str();
        let allowed = url.scheme() == "tauri"
            || (url.scheme() == "http"
                && url.host_str() == Some("127.0.0.1")
                && url.port().map(|port| format!("http://127.0.0.1:{port}"))
                    == Some(backend_origin.clone()));
        if allowed {
            true
        } else {
            let _ = open::that(target);
            false
        }
    }
}
