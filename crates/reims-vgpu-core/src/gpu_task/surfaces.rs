//! Host round trips: the IOSurfaces the host lends the task, cached by mapping id with a
//! count, and the shared-texture handles.
//!
//! A retain that misses holds the cache's lock across its round trip, so a second retain of
//! any surface waits for it. A release checks the id and the surface, and the release that
//! ends a surface's count drops the cache's reference and tells the host under the lock:
//! the host's own count of the surface decides whether the guest may unmap it.

use std::sync::Arc;

use parking_lot::{Condvar, Mutex};

use super::DeathKind;
use super::service::GpuTask;
use crate::fatal;
use crate::platform::os::Clock;
use crate::platform::{Handle, Platform};
use crate::tasks::isolation::TaskLife;
use crate::tasks::wire::{Reply, ToHost};

/// A surface the host lent, and how many retains it has.
pub(super) struct CachedSurface<P: Platform> {
    /// The cache's reference.
    pub(super) surface: P::Surface,
    /// The retains not yet released.
    pub(super) count: u64,
}

/// A slot a round trip's reply fills once, and its waiter blocks on.
struct ReplySlot<T> {
    value: Mutex<Option<T>>,
    filled: Condvar,
}

impl<T: Send + 'static> ReplySlot<T> {
    fn new() -> Arc<ReplySlot<T>> {
        Arc::new(ReplySlot {
            value: Mutex::new(None),
            filled: Condvar::new(),
        })
    }

    /// Fills the slot with the host's reply (the first one: the host replies once).
    fn fill(&self, v: T) {
        let mut g = self.value.lock();
        if g.is_none() {
            *g = Some(v);
        }
        drop(g);
        self.filled.notify_all();
    }

    /// Blocks until the reply, however long the host takes, or until the task ends: a task
    /// that died elsewhere, or that the host invalidated, stops waiting. A thread running the
    /// task's work then stops here (`TaskLife::stop_if_ended`); `None` for any other.
    fn wait<P: Platform>(self: &Arc<Self>, platform: &P, life: &Arc<TaskLife>) -> Option<T> {
        let slot = Arc::clone(self);
        let wake = move || {
            let _g = slot.value.lock();
            slot.filled.notify_all();
        };
        let reply = life.waiting(wake, || {
            let mut g = self.value.lock();
            loop {
                if let Some(v) = g.take() {
                    return Some(v);
                }
                if life.has_ended() {
                    return None;
                }
                Clock::wait(platform.clock(), &self.filled, &mut g);
            }
        });
        if reply.is_none() {
            life.stop_if_ended();
        }
        reply
    }
}

impl<P: Platform> GpuTask<P> {
    /// The surface the host lends for `identifier`, another reference: from the cache, one
    /// more retain, or from the host (under the cache's lock), cached with one retain. `None`
    /// (logged) when the host has none; it is not cached.
    pub(super) fn retain_surface(&self, identifier: u64) -> Option<P::Surface> {
        let mut surfaces = self.surfaces.lock();
        if let Some(cached) = surfaces.get_mut(&identifier) {
            cached.count = cached.count.wrapping_add(1);
            return Some(cached.surface.clone_ref());
        }
        let slot: Arc<ReplySlot<Option<P::Surface>>> = ReplySlot::new();
        let fill = Arc::clone(&slot);
        // The reply runs on the connection's queue while this one waits holding the cache
        self.send_host(ToHost::RetainIoSurface {
            mapping_id: identifier,
            reply: Reply::new(move |reply: Option<P::Surface>| {
                if reply.is_none() {
                    log::warn!("the host has no surface of mapping id {identifier}");
                }
                fill.fill(reply);
            }),
        });
        let surface = slot.wait(&*self.platform, &self.life).flatten();
        if let Some(s) = &surface
            && !surfaces.contains_key(&identifier)
        {
            surfaces.insert(
                identifier,
                CachedSurface {
                    surface: s.clone_ref(),
                    count: 1,
                },
            );
        }
        // The caller's reference; the reply's goes
        surface.as_ref().map(Handle::clone_ref)
    }

    /// Releases one retain of `identifier`'s `surface`. An id the cache does not hold, or a
    /// surface that is not the cached one, ends the task (logged). The last release drops
    /// the cache's reference and tells the host, under the cache's lock.
    pub(super) fn release_surface(&self, identifier: u64, surface: Option<P::Surface>) {
        let mut surfaces = self.surfaces.lock();
        let Some(cached) = surfaces.get_mut(&identifier) else {
            log::error!(
                "task {}: surface of mapping id {identifier} released, but none is kept",
                self.id()
            );
            fatal!(DeathKind::UnknownSurface)
        };
        // By identity
        if surface.as_ref().map(Handle::id) != Some(cached.surface.id()) {
            log::error!(
                "task {}: surface of mapping id {identifier} released is not the one kept",
                self.id()
            );
            fatal!(DeathKind::WrongSurface)
        }
        drop(surface);
        cached.count = cached.count.wrapping_sub(1);
        if cached.count == 0 {
            let released = surfaces.remove(&identifier);
            drop(released);
            self.send_host(ToHost::ReleaseIoSurface {
                mapping_id: identifier,
            });
        }
    }

    /// The host's handle for shared texture `identifier`; `None` (logged) when it has none.
    pub(super) fn new_shared_texture_handle(
        &self,
        identifier: u32,
    ) -> Option<P::SharedTextureHandle> {
        let slot: Arc<ReplySlot<Option<P::SharedTextureHandle>>> = ReplySlot::new();
        let fill = Arc::clone(&slot);
        self.send_host(ToHost::NewSharedTextureHandle {
            id: identifier,
            reply: Reply::new(move |reply: Option<P::SharedTextureHandle>| {
                fill.fill(reply);
            }),
        });
        let handle = slot.wait(&*self.platform, &self.life).flatten();
        if handle.is_none() {
            log::warn!(
                "task {}: the host has no handle for shared texture {identifier}",
                self.id()
            );
        }
        handle
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;
    use crate::testkit::capture::catch_death;
    use crate::testkit::objects::{Fk, Model, SurfaceModel, kind};

    fn surface(p: &F) -> Fk<kind::Surface> {
        p.world()
            .new_obj::<kind::Surface>(Model::Surface(SurfaceModel::default()))
    }

    /// One round trip per surface; the last release tells the host
    #[test]
    fn surfaces_are_cached_and_counted() {
        let tx = fixture(1, &rw_pages(4));
        let s = surface(&tx.p);
        *tx.host.surface.lock() = Some(s.clone_ref());
        let a = tx.task.retain_surface(5).unwrap();
        let b = tx.task.retain_surface(5).unwrap();
        assert_eq!(a.id(), s.id());
        assert_eq!(*tx.host.sent.lock(), vec!["retain 5".to_owned()]);
        tx.task.release_surface(5, Some(a));
        assert_eq!(tx.host.sent.lock().len(), 1);
        tx.task.release_surface(5, Some(b));
        assert_eq!(
            *tx.host.sent.lock(),
            vec!["retain 5".to_owned(), "release 5".to_owned()]
        );
        assert!(tx.task.surfaces.lock().is_empty());
    }

    /// A retain the host never answers waits until the task ends, then stops its thread
    /// without caching anything
    #[test]
    fn a_retain_the_host_never_answers_stops_at_the_end() {
        #![allow(clippy::disallowed_methods)] // the test polls in real time
        let tx = fixture(1, &rw_pages(4));
        tx.host
            .silent
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let (task, life) = (tx.task.clone(), tx.life.clone());
        let waiter = std::thread::spawn(move || life.run(|| task.retain_surface(5).is_some()));
        let start = std::time::Instant::now();
        while tx.host.sent.lock().is_empty() {
            assert!(start.elapsed() < std::time::Duration::from_secs(10));
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(!waiter.is_finished());
        assert!(tx.life.exit());
        assert_eq!(waiter.join().unwrap(), None);
        assert!(tx.task.surfaces.lock().is_empty());
        assert!(tx.life.waiting(|| {}, || true));
    }

    /// A nil surface from the host is not cached
    #[test]
    fn a_nil_surface_is_not_cached() {
        let tx = fixture(1, &rw_pages(4));
        assert!(tx.task.retain_surface(5).is_none());
        assert!(tx.task.surfaces.lock().is_empty());
    }

    /// A release of an unknown id, or of another surface, ends the task
    #[test]
    fn bad_releases_die() {
        let tx = fixture(1, &rw_pages(4));
        assert_eq!(
            catch_death(|| tx.task.release_surface(9, None)).unwrap_err(),
            DeathKind::UnknownSurface.site()
        );
        *tx.host.surface.lock() = Some(surface(&tx.p));
        let _held = tx.task.retain_surface(5).unwrap();
        let other = surface(&tx.p);
        assert_eq!(
            catch_death(|| tx.task.release_surface(5, Some(other))).unwrap_err(),
            DeathKind::WrongSurface.site()
        );
    }

    /// A shared texture handle comes from the host, or none
    #[test]
    fn shared_texture_handle_round_trip() {
        let tx = fixture(1, &rw_pages(4));
        assert!(tx.task.new_shared_texture_handle(3).is_none());
        let h =
            tx.p.world()
                .new_obj::<kind::SharedTextureHandle>(Model::None);
        *tx.host.handle.lock() = Some(h.clone_ref());
        assert_eq!(tx.task.new_shared_texture_handle(4).unwrap().id(), h.id());
        assert_eq!(
            *tx.host.sent.lock(),
            vec!["handle 3".to_owned(), "handle 4".to_owned()]
        );
    }
}
