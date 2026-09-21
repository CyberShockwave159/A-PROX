use std::sync::Mutex;
use sysinfo::System;

pub struct SystemWatchdog {
    system: Mutex<System>,
    min_free_ram_gb: f64,
}

impl SystemWatchdog {
    pub fn new(min_free_ram_gb: f64) -> Self {
        let mut sys = System::new_all();
        sys.refresh_memory();
        Self {
            system: Mutex::new(sys),
            min_free_ram_gb,
        }
    }

    /// Checks if the host has adequate RAM headroom to safely execute operations
    pub fn is_memory_safe(&self) -> (bool, f64) {
        let mut sys = self.system.lock().unwrap();
        sys.refresh_memory();

        let available_bytes = sys.available_memory();
        let available_gb = available_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

        (available_gb >= self.min_free_ram_gb, available_gb)
    }

    pub fn get_telemetry(&self) -> (f64, f64, f32) {
        let mut sys = self.system.lock().unwrap();
        sys.refresh_memory();
        sys.refresh_cpu_all();

        let total_gb = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        let available_gb = sys.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        let global_cpu = sys.global_cpu_usage();

        (total_gb, available_gb, global_cpu)
    }
}
