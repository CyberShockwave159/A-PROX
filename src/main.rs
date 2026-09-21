use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use a_prox::{config::AppConfig, server, state::AppState};

#[derive(Parser, Debug)]
#[command(name = "a-prox")]
#[command(about = "Hardware-Optimized LLM Middleware Proxy for llama.cpp", long_about = None)]
struct Cli {
    #[arg(short, long, default_value = "config/default.toml")]
    config: PathBuf,

    #[arg(short, long)]
    port: Option<u16>,

    #[arg(long)]
    host: Option<String>,

    #[arg(long)]
    upstream: Option<String>,

    #[arg(long)]
    searxng_port: Option<u16>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,a_prox=debug".into()),
        )
        .init();

    let cli = Cli::parse();
    let mut config = AppConfig::load(&cli.config)?;

    if let Some(p) = cli.port {
        config.server.port = p;
    }
    if let Some(h) = cli.host {
        config.server.host = h;
    }
    if let Some(u) = cli.upstream {
        config.upstream.base_url = u;
    }
    if let Some(p) = cli.searxng_port {
        config.searxng.port = p;
        config.searxng.listen_port = p;
    }

    print_banner(&config);

    let state = Arc::new(AppState::new(config).await?);

    server::run_server(state).await?;

    Ok(())
}

fn print_banner(config: &AppConfig) {
    eprintln!(r#"
   ___      ____  ____  ____ _  __
  / _ |____/ __ \/ __ \/ __ \ |/ /
 / __ /___/ /_/ / /_/ / /_/ /   / 
/_/ |_|  / .___/_/   \____//_/|_|  
        /_/                       
Hardware-Optimized LLM API Middleware Proxy
============================================================
 Host GPU:          RTX 3080 Ti (12GB) -> STRICT ZERO VRAM (0 MB)
 Host CPU:          AMD Ryzen 9 5900X (12C/24T) -> Pinning CCD1
 System RAM:        128 GB DDR4 -> 16GB MMAP + 1GB Cache
 Upstream Engine:   {} (Key: protected)
 Proxy Gateway:     http://{}:{}
============================================================
"#,
        config.upstream.base_url,
        config.server.host,
        config.server.port
    );
}
