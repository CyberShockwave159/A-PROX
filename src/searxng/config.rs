use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct SearXNGSettings {
    pub general: General,
    pub search: Search,
    pub server: Server,
    pub engines: Vec<EngineConfig>,
    pub outgoing: Outgoing,
}

#[derive(Debug, Serialize)]
pub struct General {
    pub debug: bool,
    #[serde(rename = "instance_name")]
    pub instance_name: String,
}

#[derive(Debug, Serialize)]
pub struct Search {
    pub safe_search: u8,
    pub autocomplete: String,
    pub default_lang: String,
    pub ban_time_between_queries: u32,
    pub formats: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct Server {
    pub port: u16,
    pub bind_address: String,
    #[serde(rename = "secret_key")]
    pub secret_key: String,
    pub limiter: bool,
    pub http_protocol_version: String,
    pub image_proxy: bool,
    pub method: String,
    pub default_http_headers: DefaultHttpHeaders,
}

#[derive(Debug, Serialize)]
pub struct DefaultHttpHeaders {
    #[serde(rename = "X-Content-Type-Options")]
    pub x_content_type_options: String,
}

#[derive(Debug, Serialize)]
pub struct Outgoing {
    pub request_timeout: f64,
    pub max_request_timeout: f64,
    pub useragent_suffix: String,
    pub max_retries: u8,
}

#[derive(Debug, Serialize)]
pub struct EngineConfig {
    pub name: String,
    pub engine: String,
    #[serde(rename = "shortcut")]
    pub shortcut: String,
    pub disabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paging: Option<bool>,
}

pub fn generate_settings(config: &crate::config::SearXNGConfig, port: u16) -> String {
    let engines: Vec<EngineConfig> = config
        .search_engines
        .split(',')
        .map(|name| {
            let name = name.trim();
            let shortcut = name.chars().take(2).collect::<String>().to_lowercase();
            let engine = name.to_string();
            let mut cfg = EngineConfig {
                name: name.to_string(),
                engine: engine.clone(),
                shortcut,
                disabled: false,
                search_type: None,
                paging: Some(true),
            };
            // Wikipedia has auto search_type
            if name == "wikipedia" {
                cfg.search_type = Some("auto".to_string());
            }
            cfg
        })
        .collect();

    let settings = SearXNGSettings {
        general: General {
            debug: false,
            instance_name: "A-PROX SearXNG".to_string(),
        },
        search: Search {
            safe_search: 0,
            autocomplete: String::new(),
            default_lang: String::new(),
            ban_time_between_queries: 0,
            formats: vec!["html".to_string(), "json".to_string()],
        },
        server: Server {
            port,
            bind_address: "127.0.0.1".to_string(),
            secret_key: "a-prox-searxng-secret-key".to_string(),
            limiter: false,
            http_protocol_version: "1.1".to_string(),
            image_proxy: false,
            method: "GET".to_string(),
            default_http_headers: DefaultHttpHeaders {
                x_content_type_options: "nosniff".to_string(),
            },
        },
        engines,
        outgoing: Outgoing {
            request_timeout: 5.0,
            max_request_timeout: 15.0,
            useragent_suffix: "A-PROX".to_string(),
            max_retries: 1,
        },
    };

    serde_yaml::to_string(&settings).expect("Failed to serialize SearXNG settings")
}
