//! Detached Windows threads with the stack and name core requests.

use super::clock::WindowsClock;
use reims_vgpu_core::platform::os::{Threads, TimerClock};
use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal};
use std::sync::Arc;

/// The process's thread service.
#[derive(Debug, Default)]
pub struct WindowsThreads;
impl WindowsThreads {
    /// The process's threads.
    pub fn new() -> Self {
        Self
    }
}
impl Threads for WindowsThreads {
    fn spawn(&self, name: &'static str, stack_size: usize, body: Box<dyn FnOnce() + Send>) {
        if std::thread::Builder::new()
            .name(name.to_owned())
            .stack_size(stack_size)
            .spawn(body)
            .is_err()
        {
            fatal(DeathId::Platform {
                kind: DeathKind::Abort,
                call: "CreateThread",
            });
        }
    }
    fn spawn_worker(&self, label: &'static str, body: Box<dyn FnOnce() + Send>) {
        self.spawn(label, reims_vgpu_core::rt::queue::WORKER_STACK, body);
    }
    fn handle(&self) -> Arc<dyn Threads> {
        Arc::new(Self)
    }
    fn timer_clock(&self) -> Option<Arc<dyn TimerClock>> {
        Some(Arc::new(WindowsClock))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn workers_keep_the_requested_name() {
        let (send, receive) = std::sync::mpsc::channel();
        WindowsThreads.spawn_worker(
            "reims-vgpu.ui",
            Box::new(move || {
                send.send(std::thread::current().name().map(str::to_owned))
                    .unwrap();
            }),
        );
        assert_eq!(
            receive
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap()
                .as_deref(),
            Some("reims-vgpu.ui")
        );
    }
}
