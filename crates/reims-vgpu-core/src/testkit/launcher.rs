//! The test launcher, [`ThreadLauncher`]: the product's launcher (`tasks::isolation::
//! TaskThreads`) over the fake platform, recording each task it launched and letting a test
//! script the service.
//!
//! A launched task shares the launching platform, as an isolated task shares its process: its
//! objects come from the same fake device and world, and it reads guest RAM through the same fake
//! address space. A death in the task's own code ends the task, never the test; a scripted crash
//! inside Metal (`Step::crash`) ends the test's "process", as it ends the real one.

use std::sync::Arc;

use super::platform::FakePlatform;
use crate::raise::death::{DeathId, DeathSite};
use crate::sync::Mutex;
use crate::tasks::isolation::{TaskLife, TaskThreads};
use crate::tasks::launcher::{HostLink, HostSide, ServiceLink, ServiceSide, TaskLauncher};

/// Makes the service of one launch from the launching platform and the service's end of the
/// connection: `GpuTask::new` in the real wiring.
pub type ServiceFactory = Arc<
    dyn Fn(Arc<FakePlatform>, Arc<dyn HostLink<FakePlatform>>) -> Arc<dyn ServiceSide<FakePlatform>>
        + Send
        + Sync,
>;

/// What a test can observe of one launched task.
#[derive(Debug)]
pub struct Launched {
    /// The task's life.
    pub life: Arc<TaskLife>,
}

impl Launched {
    /// Whether the task has ended (died, or invalidated by the host).
    pub fn ended(&self) -> bool {
        self.life.has_ended()
    }

    /// Where the task died, if a death in its own code ended it.
    pub fn death(&self) -> Option<DeathSite> {
        self.life.death().map(DeathId::site)
    }
}

/// The test launcher: the product's, recording what it launched.
pub struct ThreadLauncher {
    threads: TaskThreads<FakePlatform>,
    launched: Mutex<Vec<Arc<Launched>>>,
}

impl Default for ThreadLauncher {
    fn default() -> Self {
        ThreadLauncher::new()
    }
}

impl std::fmt::Debug for ThreadLauncher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreadLauncher")
            .field("launched", &self.launched.lock().len())
            .finish_non_exhaustive()
    }
}

impl ThreadLauncher {
    /// A launcher whose services are `GpuTask`s.
    pub fn new() -> ThreadLauncher {
        ThreadLauncher {
            threads: TaskThreads::new(),
            launched: Mutex::new(Vec::new()),
        }
    }

    /// A launcher whose services `factory` makes (a unit test's scripted service).
    pub fn with_factory(factory: ServiceFactory) -> ThreadLauncher {
        ThreadLauncher {
            threads: TaskThreads::with_factory(Arc::new(move |platform, host, _life| {
                factory(Arc::clone(platform), host)
            })),
            launched: Mutex::new(Vec::new()),
        }
    }

    /// Every task launched so far, in order.
    pub fn launched(&self) -> Vec<Arc<Launched>> {
        self.launched.lock().clone()
    }
}

impl TaskLauncher<FakePlatform> for ThreadLauncher {
    fn launch(
        &self,
        platform: &Arc<FakePlatform>,
        host: Arc<dyn HostSide<FakePlatform>>,
    ) -> Arc<dyn ServiceLink<FakePlatform>> {
        let (link, life) = self.threads.launch_task(platform, host);
        self.launched.lock().push(Arc::new(Launched { life }));
        link
    }
}

/// Where a launched task died, if it did.
pub fn death_of(l: &Launched) -> Option<DeathSite> {
    l.death()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // tests poll in real time

    use super::*;
    use crate::raise::death::DeathKind;
    use crate::tasks::wire::{Reply, ToHost, ToService};
    use std::time::{Duration, Instant};

    #[derive(Default)]
    struct Host {
        events: Mutex<Vec<String>>,
    }

    impl HostSide<FakePlatform> for Host {
        fn on_message(&self, m: ToHost<FakePlatform>) {
            self.events.lock().push(m.selector().to_owned());
        }
        fn on_interrupted(&self) {
            self.events.lock().push("interrupted".into());
        }
        fn on_invalidated(&self) {
            self.events.lock().push("invalidated".into());
        }
    }

    /// Where the scripted service dies: its own check.
    const DEATH: DeathSite = DeathSite {
        kind: DeathKind::Abort,
        file: "scripted",
        line: 0,
    };

    /// FlushResources replies whether the service sees the launching platform's device;
    /// UnmapMemory dies.
    struct Scripted {
        platform: Arc<FakePlatform>,
    }

    impl ServiceSide<FakePlatform> for Scripted {
        fn on_message(&self, m: ToService<FakePlatform>) {
            match m {
                ToService::FlushResources { reply } => {
                    reply.call(u32::from(
                        self.platform.world().live("MTLDevice").len() == 1,
                    ));
                }
                ToService::UnmapMemory { .. } => crate::fatal!(DEATH),
                _ => {}
            }
        }
    }

    fn wait_until(f: impl Fn() -> bool) {
        let start = Instant::now();
        while !f() {
            assert!(start.elapsed() < Duration::from_secs(10), "timed out");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Each launch is recorded, its service made over the launching platform; a death in the
    /// service ends that task only, and the test's own slot sees nothing
    #[test]
    fn records_each_launch_and_its_death() {
        let l = ThreadLauncher::with_factory(Arc::new(|platform, _host| {
            Arc::new(Scripted { platform })
        }));
        let p = Arc::new(FakePlatform::new());
        let host = Arc::new(Host::default());
        let link = l.launch(&p, host.clone());
        let got = Arc::new(Mutex::new(Vec::new()));
        let g2 = got.clone();
        link.send(ToService::FlushResources {
            reply: Reply::new(move |v| g2.lock().push(v)),
        });
        wait_until(|| !got.lock().is_empty());
        assert_eq!(*got.lock(), vec![1]);
        let other = l.launch(&p, Arc::new(Host::default()));
        link.send(ToService::UnmapMemory {
            offset: 0,
            length: 0,
            reply: Reply::new(|_| {}),
        });
        wait_until(|| host.events.lock().iter().any(|e| e == "interrupted"));
        let launched = l.launched();
        assert_eq!(launched.len(), 2);
        assert!(launched[0].ended());
        assert_eq!(death_of(&launched[0]), Some(DEATH));
        assert!(!launched[1].ended());
        other.invalidate();
        assert!(launched[1].ended() && launched[1].death().is_none());
        assert!(crate::testkit::capture::current_slot().is_none_or(|s| s.death().is_none()));
        assert!(format!("{l:?}").contains("launched: 2"));
    }
}
