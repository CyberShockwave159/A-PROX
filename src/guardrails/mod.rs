pub mod watchdog;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
pub use watchdog::SystemWatchdog;
use crate::error::AppError;

pub struct ConcurrencyLimiter {
    semaphore: Arc<Semaphore>,
    active_permits: Arc<AtomicUsize>,
    queued_requests: Arc<AtomicUsize>,
    max_queue_depth: usize,
}

impl ConcurrencyLimiter {
    pub fn new(max_concurrent: usize, max_queue_depth: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            active_permits: Arc::new(AtomicUsize::new(0)),
            queued_requests: Arc::new(AtomicUsize::new(0)),
            max_queue_depth,
        }
    }

    /// Attempts to acquire an inference permit, queuing if necessary or rejecting if queue full
    pub async fn acquire_permit(&self) -> Result<InferencePermitGuard, AppError> {
        let current_queue = self.queued_requests.fetch_add(1, Ordering::SeqCst);

        if current_queue >= self.max_queue_depth {
            self.queued_requests.fetch_sub(1, Ordering::SeqCst);
            return Err(AppError::RateLimit(format!(
                "Inference queue full: {} requests already queued",
                self.max_queue_depth
            )));
        }

        let permit_result = self.semaphore.clone().acquire_owned().await;
        self.queued_requests.fetch_sub(1, Ordering::SeqCst);

        match permit_result {
            Ok(permit) => {
                self.active_permits.fetch_add(1, Ordering::SeqCst);
                Ok(InferencePermitGuard {
                    _permit: permit,
                    active_counter: Arc::clone(&self.active_permits),
                })
            }
            Err(_) => Err(AppError::ResourceExhausted("Concurrency semaphore closed".to_string())),
        }
    }

    pub fn stats(&self) -> (usize, usize) {
        (
            self.active_permits.load(Ordering::Relaxed),
            self.queued_requests.load(Ordering::Relaxed),
        )
    }
}

pub struct InferencePermitGuard {
    _permit: OwnedSemaphorePermit,
    active_counter: Arc<AtomicUsize>,
}

impl Drop for InferencePermitGuard {
    fn drop(&mut self) {
        self.active_counter.fetch_sub(1, Ordering::SeqCst);
    }
}
