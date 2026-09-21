use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};

#[derive(Debug, Clone, Serialize)]
pub struct ActiveRequest {
    pub request_id: u64,
    pub route_decision: String,
    pub start_time: u64,
    pub messages_count: usize,
    pub tokens_sent: usize,
    pub tokens_received: usize,
    pub tool_calls_executed: Vec<String>,
    pub rag_hits: usize,
    pub upstream_latency_ms: Option<f64>,
    pub is_streaming: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RequestHistoryEntry {
    pub request_id: u64,
    pub timestamp: u64,
    pub route_decision: String,
    pub status: String,
    pub duration_ms: f64,
    pub messages_count: usize,
    pub tokens_sent: usize,
    pub tokens_received: usize,
    pub tool_calls_executed: Vec<String>,
    pub rag_hits: usize,
    pub upstream_latency_ms: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DbStats {
    pub total_chunks: i64,
    pub collections: HashMap<String, i64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearXNGStatus {
    pub enabled: bool,
    pub running: bool,
    pub listen_port: u16,
}

#[derive(Debug, Clone, Serialize)]
pub struct MonitorApiData {
    pub system: SystemInfo,
    pub concurrency: ConcurrencyInfo,
    pub active_requests: Vec<ActiveRequest>,
    pub history: Vec<RequestHistoryEntry>,
    pub db_stats: DbStats,
    pub searxng: SearXNGStatus,
    pub config: MonitorConfigInfo,
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemInfo {
    pub total_ram_gb: f64,
    pub available_ram_gb: f64,
    pub used_ram_gb: f64,
    pub ram_pct: f64,
    pub global_cpu_usage_pct: f32,
    pub uptime_seconds: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConcurrencyInfo {
    pub active_inferences: usize,
    pub queued_requests: usize,
    pub max_slots: usize,
    pub queue_depth: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MonitorConfigInfo {
    pub upstream_base_url: String,
    pub upstream_model_alias: String,
    pub upstream_timeout_seconds: u64,
    pub max_context_tokens: usize,
    pub embedding_dimension: usize,
    pub db_path: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MonitorStatistics {
    pub total_requests: usize,
    pub success_count: usize,
    pub error_count: usize,
    pub success_rate_pct: f64,
    pub route_distribution: Vec<RouteDistributionEntry>,
    pub tool_usage: Vec<ToolUsageEntry>,
    pub latency: LatencyStats,
    pub token_stats: TokenStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouteDistributionEntry {
    pub route: String,
    pub count: usize,
    pub success_count: usize,
    pub error_count: usize,
    pub avg_duration_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolUsageEntry {
    pub tool_name: String,
    pub total_calls: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencyStats {
    pub avg_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenStats {
    pub total_tokens_sent: usize,
    pub total_tokens_received: usize,
    pub avg_tokens_sent: usize,
    pub avg_tokens_received: usize,
    pub total_token_throughput_per_sec: f64,
}

#[derive(Debug, Clone)]
pub struct MonitorState {
    pub next_request_id: Arc<Mutex<u64>>,
    pub active_requests: Arc<Mutex<Vec<ActiveRequest>>>,
    pub history: Arc<Mutex<Vec<RequestHistoryEntry>>>,
    pub broadcast_sender: broadcast::Sender<String>,
    pub max_history: usize,
    pub start_time: u64,
}

impl MonitorState {
    pub fn new(max_history: usize) -> Self {
        let (sender, _) = broadcast::channel(128);
        Self {
            next_request_id: Arc::new(Mutex::new(1)),
            active_requests: Arc::new(Mutex::new(Vec::new())),
            history: Arc::new(Mutex::new(Vec::new())),
            broadcast_sender: sender,
            max_history,
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    pub async fn next_request_id(&self) -> u64 {
        let mut guard = self.next_request_id.lock().await;
        let id = *guard;
        *guard += 1;
        id
    }

    pub async fn set_active(&self, request: ActiveRequest) {
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let req_id = request.request_id;
        let route = request.route_decision.clone();
        let msgs = request.messages_count;
        let sent = request.tokens_sent;
        let recv = request.tokens_received;
        let tools = request.tool_calls_executed.clone();
        let rag = request.rag_hits;
        let lat = request.upstream_latency_ms;

        self.active_requests.lock().await.push(request);

        {
            let mut history = self.history.lock().await;
            if history.len() < 200 {
                history.push(RequestHistoryEntry {
                    request_id: req_id,
                    timestamp: now_secs,
                    route_decision: route,
                    status: "processing".to_string(),
                    duration_ms: 0.0,
                    messages_count: msgs,
                    tokens_sent: sent,
                    tokens_received: recv,
                    tool_calls_executed: tools,
                    rag_hits: rag,
                    upstream_latency_ms: lat,
                    error: None,
                });
                while history.len() > self.max_history {
                    history.remove(0);
                }
            }
        }

        self.broadcast("active_requests_updated".to_string());
    }

    pub async fn complete_request(&self, request_id: u64, status: &str, duration_ms: f64, error: Option<String>) {
        let mut active = self.active_requests.lock().await;
        let update_data = active.iter().find(|r| r.request_id == request_id).cloned();
        active.retain(|r| r.request_id != request_id);

        let mut history = self.history.lock().await;
        if let Some(entry) = history.iter_mut().find(|e| e.request_id == request_id) {
            // Only finalize an in-progress entry. This keeps completion idempotent so a
            // late/streaming completion can never be clobbered by an earlier call.
            if entry.status == "processing" {
                entry.status = status.to_string();
                entry.duration_ms = duration_ms;
                entry.error = error;

                // Propagate tool calls and upstream latency from active request data
                if let Some(active_data) = &update_data {
                    if !active_data.tool_calls_executed.is_empty() {
                        entry.tool_calls_executed = active_data.tool_calls_executed.clone();
                    }
                    entry.upstream_latency_ms = active_data.upstream_latency_ms;
                    entry.tokens_received = active_data.tokens_received;
                    entry.tokens_sent = active_data.tokens_sent;
                    entry.rag_hits = active_data.rag_hits;
                }
            }
        }
        while history.len() > self.max_history {
            history.remove(0);
        }

        self.broadcast("request_completed".to_string());
    }

    pub async fn update_active_request(
        &self,
        request_id: u64,
        tokens_received: usize,
        tool_calls_executed: Vec<String>,
        rag_hits: usize,
        upstream_latency_ms: Option<f64>,
    ) {
        let mut active = self.active_requests.lock().await;
        if let Some(req) = active.iter_mut().find(|r| r.request_id == request_id) {
            req.tokens_received = tokens_received;
            if !tool_calls_executed.is_empty() {
                req.tool_calls_executed = tool_calls_executed;
            }
            req.rag_hits = rag_hits;
            req.upstream_latency_ms = upstream_latency_ms;
        }

        self.broadcast("active_request_updated".to_string());
    }

    /// Update only the RAG hit count, leaving token/tool/latency fields untouched.
    pub async fn update_rag_hits(&self, request_id: u64, rag_hits: usize) {
        let mut active = self.active_requests.lock().await;
        if let Some(req) = active.iter_mut().find(|r| r.request_id == request_id) {
            req.rag_hits = rag_hits;
        }
        drop(active);
        self.broadcast("active_request_updated".to_string());
    }

    /// Record upstream response metrics for pass-through / RAG routes, which do not
    /// execute tools. Leaves tool_calls_executed and rag_hits untouched.
    pub async fn set_upstream_metrics(
        &self,
        request_id: u64,
        tokens_received: usize,
        upstream_latency_ms: Option<f64>,
    ) {
        let mut active = self.active_requests.lock().await;
        if let Some(req) = active.iter_mut().find(|r| r.request_id == request_id) {
            req.tokens_received = tokens_received;
            req.upstream_latency_ms = upstream_latency_ms;
        }
        drop(active);
        self.broadcast("active_request_updated".to_string());
    }

    /// Finalize a streaming pass-through request from a synchronous context (stream
    /// poll/drop). Uses try_lock so it never blocks; token metrics are recorded on the
    /// active entry and then propagated to history by `complete_request`.
    pub fn finalize_stream_sync(
        &self,
        request_id: u64,
        tokens_received: usize,
        upstream_latency_ms: Option<f64>,
        status: &str,
        duration_ms: f64,
        error: Option<String>,
    ) {
        let updated_metrics = {
            match self.active_requests.try_lock() {
                Ok(mut active) => {
                    if let Some(req) = active.iter_mut().find(|r| r.request_id == request_id) {
                        req.tokens_received = tokens_received;
                        req.upstream_latency_ms = upstream_latency_ms;
                    }
                    active.retain(|r| r.request_id != request_id);
                    true
                }
                Err(_) => false,
            }
        };

        if let Ok(mut history) = self.history.try_lock() {
            if let Some(entry) = history.iter_mut().find(|e| e.request_id == request_id) {
                if entry.status == "processing" {
                    entry.status = status.to_string();
                    entry.duration_ms = duration_ms;
                    entry.error = error;
                    if updated_metrics {
                        entry.tokens_received = tokens_received;
                        entry.upstream_latency_ms = upstream_latency_ms;
                    }
                }
            }
            while history.len() > self.max_history {
                history.remove(0);
            }
        }

        self.broadcast("request_completed".to_string());
    }

    pub async fn get_api_data(
        &self,
        watchdog: &crate::guardrails::SystemWatchdog,
        concurrency_limiter: &crate::guardrails::ConcurrencyLimiter,
        db_stats: DbStats,
        searxng_status: SearXNGStatus,
        config: &crate::config::AppConfig,
    ) -> MonitorApiData {
        let (total_ram, free_ram, cpu_usage) = watchdog.get_telemetry();
        let used_ram = total_ram - free_ram;
        let ram_pct = if total_ram > 0.0 { (used_ram / total_ram) * 100.0 } else { 0.0 };

        let (active_inferences, queued) = concurrency_limiter.stats();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        MonitorApiData {
            system: SystemInfo {
                total_ram_gb: total_ram,
                available_ram_gb: free_ram,
                used_ram_gb: used_ram,
                ram_pct,
                global_cpu_usage_pct: cpu_usage,
                uptime_seconds: now - self.start_time,
            },
            concurrency: ConcurrencyInfo {
                active_inferences,
                queued_requests: queued,
                max_slots: config.guardrails.max_concurrent_inferences,
                queue_depth: config.guardrails.queue_depth,
            },
            active_requests: self.active_requests.lock().await.clone(),
            history: self.history.lock().await.clone(),
            db_stats,
            searxng: searxng_status,
            config: MonitorConfigInfo {
                upstream_base_url: config.upstream.base_url.clone(),
                upstream_model_alias: config.upstream.model_alias.clone(),
                upstream_timeout_seconds: config.upstream.timeout_seconds,
                max_context_tokens: config.context.max_context_tokens,
                embedding_dimension: config.embeddings.dimension,
                db_path: config.db.path.clone(),
            },
        }
    }

    fn broadcast(&self, event: String) {
        let _ = self.broadcast_sender.send(event);
    }

    pub fn broadcast_receiver(&self) -> broadcast::Receiver<String> {
        self.broadcast_sender.subscribe()
    }

    pub fn compute_statistics(&self) -> MonitorStatistics {
        let completed: Vec<RequestHistoryEntry> = match self.history.try_lock() {
            Ok(guard) => guard.iter().filter(|e| e.status == "success" || e.status == "error").cloned().collect(),
            Err(_) => Vec::new(),
        };

        let total = completed.len();
        let success_count = completed.iter().filter(|e| e.status == "success").count();
        let error_count = completed.iter().filter(|e| e.status == "error").count();
        let success_rate = if total > 0 { success_count as f64 / total as f64 * 100.0 } else { 0.0 };

        let mut route_counts: HashMap<String, usize> = HashMap::new();
        let mut route_success: HashMap<String, usize> = HashMap::new();
        let mut route_errors: HashMap<String, usize> = HashMap::new();
        let mut route_durations: HashMap<String, f64> = HashMap::new();
        let mut route_dur_count: HashMap<String, usize> = HashMap::new();

        for entry in &completed {
            let route = entry.route_decision.clone();
            *route_counts.entry(route.clone()).or_insert(0) += 1;
            if entry.status == "success" {
                *route_success.entry(route.clone()).or_insert(0) += 1;
            }
            if entry.status == "error" {
                *route_errors.entry(route.clone()).or_insert(0) += 1;
            }
            *route_durations.entry(route.clone()).or_insert(0.0) += entry.duration_ms;
            *route_dur_count.entry(route).or_insert(0) += 1;
        }

        let mut route_distribution = route_counts.into_iter().map(|(route, count)| {
            let avg_dur = if let Some(&total_dur) = route_durations.get(&route) {
                total_dur / route_dur_count.get(&route).copied().unwrap_or(1) as f64
            } else { 0.0 };
            let sc = *route_success.get(&route).unwrap_or(&0);
            let ec = *route_errors.get(&route).unwrap_or(&0);
            RouteDistributionEntry {
                route,
                count,
                success_count: sc,
                error_count: ec,
                avg_duration_ms: avg_dur,
            }
        }).collect::<Vec<_>>();

        // Sort by request count (descending) so the busiest route is always first.
        // HashMap iteration order is nondeterministic, which previously made the
        // Per-Route Stats list jump around on every refresh. Tie-break by name for stability.
        route_distribution.sort_by(|a, b| {
            b.count.cmp(&a.count).then_with(|| a.route.cmp(&b.route))
        });

        let mut tool_counts: HashMap<String, usize> = HashMap::new();
        for entry in &completed {
            for tool in &entry.tool_calls_executed {
                *tool_counts.entry(tool.clone()).or_insert(0) += 1;
            }
        }
        let mut tool_usage: Vec<ToolUsageEntry> = tool_counts.into_iter().map(|(name, count)| ToolUsageEntry {
            tool_name: name,
            total_calls: count,
        }).collect();
        // Alphabetical by default, but pin the single most-used tool to the top.
        tool_usage.sort_by(|a, b| a.tool_name.cmp(&b.tool_name));
        if let Some(most_used) = tool_usage.iter().max_by_key(|e| e.total_calls).map(|e| e.tool_name.clone()) {
            if let Some(idx) = tool_usage.iter().position(|e| e.tool_name == most_used) {
                let picked = tool_usage.remove(idx);
                tool_usage.insert(0, picked);
            }
        }

        let latencies: Vec<f64> = completed.iter().map(|e| e.duration_ms).collect();
        let latency_stats = if !latencies.is_empty() {
            let mut sorted = latencies.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let avg = sorted.iter().sum::<f64>() / sorted.len() as f64;
            let p50 = Self::percentile(&sorted, 50.0);
            let p95 = Self::percentile(&sorted, 95.0);
            let p99 = Self::percentile(&sorted, 99.0);
            LatencyStats {
                avg_ms: avg, p50_ms: p50, p95_ms: p95, p99_ms: p99,
                min_ms: sorted[0], max_ms: sorted[sorted.len() - 1], total_ms: sorted.iter().sum(),
            }
        } else {
            LatencyStats { avg_ms: 0.0, p50_ms: 0.0, p95_ms: 0.0, p99_ms: 0.0, min_ms: 0.0, max_ms: 0.0, total_ms: 0.0 }
        };

        let total_sent: usize = completed.iter().map(|e| e.tokens_sent).sum();
        let total_received: usize = completed.iter().map(|e| e.tokens_received).sum();
        let total_time: f64 = completed.iter()
            .filter(|e| e.duration_ms > 0.0)
            .map(|e| e.duration_ms / 1000.0)
            .sum();
        let tokens_per_sec = if total_time > 0.0 { (total_sent + total_received) as f64 / total_time } else { 0.0 };

        MonitorStatistics {
            total_requests: total,
            success_count,
            error_count,
            success_rate_pct: success_rate,
            route_distribution,
            tool_usage,
            latency: latency_stats,
            token_stats: TokenStats {
                total_tokens_sent: total_sent,
                total_tokens_received: total_received,
                avg_tokens_sent: if total > 0 { total_sent / total } else { 0 },
                avg_tokens_received: if total > 0 { total_received / total } else { 0 },
                total_token_throughput_per_sec: tokens_per_sec,
            },
        }
    }

    pub fn percentile(sorted: &[f64], p: f64) -> f64 {
        if sorted.is_empty() { return 0.0; }
        let index = (p / 100.0 * sorted.len() as f64) as usize;
        let index = index.min(sorted.len() - 1);
        sorted[index]
    }

    pub fn heartbeat(&self) {
        let event = serde_json::json!({
            "type": "heartbeat",
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        });
        let _ = self.broadcast_sender.send(event.to_string());
    }
}
