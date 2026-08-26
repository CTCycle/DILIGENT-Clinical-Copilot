use std::sync::{Arc, Mutex};

use tauri::Url;

pub fn navigation_handler(
    backend_origin: Arc<Mutex<Option<String>>>,
) -> impl Fn(&Url) -> bool + Send + 'static {
    move |url| {
        let target = url.as_str();
        let local_app = (url.scheme() == "tauri" && url.host_str() == Some("localhost"))
            || (matches!(url.scheme(), "http" | "https")
                && url.host_str() == Some("tauri.localhost")
                && url.port().is_none());
        let configured_origin = backend_origin.lock().ok().and_then(|value| value.clone());
        let backend_page = configured_origin.is_some_and(|origin| {
            target == origin
                || target.starts_with(&format!("{origin}/"))
                || target.starts_with(&format!("{origin}?"))
                || target.starts_with(&format!("{origin}#"))
        });
        let allowed = local_app || backend_page;
        if allowed {
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
    use std::sync::{Arc, Mutex};
    use tauri::Url;

    #[test]
    fn allows_tauri_app_and_exact_backend_origins() {
        let backend = Arc::new(Mutex::new(Some("http://127.0.0.1:48123".to_string())));
        let handler = navigation_handler(backend);

        assert!(handler(&Url::parse("http://tauri.localhost/").unwrap()));
        assert!(handler(
            &Url::parse("http://127.0.0.1:48123/clinical-sessions").unwrap()
        ));
        assert!(!handler(&Url::parse("file:///C:/outside.html").unwrap()));
    }
}
