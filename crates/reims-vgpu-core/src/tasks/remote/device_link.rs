//! What the service asks of the host's device: the IOSurfaces the host keeps alive for it, by
//! mapping id, and the kernel task's shared textures.
//!
//! The device is held unretained ([`TaskDeviceLink`]); a device that is gone answers like a nil
//! one (no mapper service, no kernel task). The device owns its tasks, which own their
//! endpoints, so a live task never meets that case.
//!
//! Ordering contracts (each named test pins it):
//! - **A surface's retain** looks the surface up through the device's mapper service, keeps one
//!   reference under its mapping id unless the id already has one (then it logs), and replies
//!   with the surface, all under the endpoint's lock
//!   (`a_second_retain_of_one_id_keeps_the_entry_and_logs`).
//!   An id the mapper service does not know is kept as a nil entry
//!   (`no_mapper_service_replies_nil_and_keeps_a_nil_entry`).
//! - **A surface's release** drops its entry without the endpoint's lock
//!   (`releases_without_the_endpoint_lock`).
//! - **A shared-texture handle** comes from the kernel task, which the hold's holder keeps to
//!   its end; a raise on the way is logged and the reply is nil
//!   (`a_raise_from_the_kernel_task_is_logged_and_replies_nil`).

use std::sync::Arc;

use parking_lot::Mutex;
use zerocopy::IntoBytes;

use crate::collections::LookupMap;
use crate::fifo::wire::Records;
use crate::num::Truncate;
use crate::platform::{Handle, Platform};
use crate::raise::{OnFault, Raise, RaiseResult};
use crate::rt::Hold;
use crate::seams::links::{KernelTaskRef, TaskDeviceLink};
use crate::tasks::wire::Reply;

/// The endpoint's link to its device.
pub struct DeviceLink<P: Platform> {
    device: TaskDeviceLink<P>,
    /// The surfaces kept alive for the service, by mapping id; `None` for an id the mapper
    /// service did not know. Its lock guards the map's memory only: the endpoint's lock is the
    /// one retains and crashes take.
    surfaces: Mutex<LookupMap<u64, Option<P::Surface>>>,
}

/// A caught raise in a log line: its detail.
fn exception(e: &Raise) -> &str {
    e.exception()
}

impl<P: Platform> DeviceLink<P> {
    /// A link to `device` with no surface kept.
    pub(crate) fn new(device: TaskDeviceLink<P>) -> DeviceLink<P> {
        DeviceLink {
            device,
            surfaces: Mutex::new(LookupMap::new()),
        }
    }

    /// The surface `mapping_id` names, through the device's mapper service: kept alive under the
    /// id unless the id already has an entry (which stays, and the retain logs), and passed to
    /// `reply`. The caller holds the endpoint's lock across it.
    pub(crate) fn retain_surface(&self, mapping_id: u64, reply: &Reply<Option<P::Surface>>) {
        // +1, nil without a mapper service.
        let surface = self
            .device
            .upgrade()
            .and_then(|d| d.mapper_service())
            .and_then(|m| m.retain_surface_with_mapping_id(mapping_id));
        let kept = {
            let mut map = self.surfaces.lock();
            if map.contains_key(&mapping_id) {
                false
            } else {
                // The map's own reference; a nil surface is kept as a nil entry.
                map.insert(mapping_id, surface.as_ref().map(Handle::clone_ref));
                true
            }
        };
        if !kept {
            log::warn!("surface of mapping id {mapping_id} retained again: the first stays kept");
        }
        // The reply takes the +1; the transport drops it once the service has its own object
        // over the surface.
        reply.call(surface);
    }

    /// Drops the surface kept under `mapping_id`, if any.
    pub(crate) fn release_surface(&self, mapping_id: u64) {
        // Erased without the endpoint's lock (only the map's own).
        let released = self.surfaces.lock().remove(&mapping_id);
        drop(released);
    }

    /// Drops every kept surface (the task died). The caller holds the endpoint's lock. The
    /// order among them is the map's, which nothing observes.
    pub(crate) fn release_all_surfaces(&self) {
        let released = std::mem::take(&mut *self.surfaces.lock());
        drop(released);
    }

    /// The kernel task, which the hold's holder keeps to its end ([`Hold::keep`]), so a
    /// delete meanwhile tears it down after this work; `None` for a device that is gone. A raise
    /// from the lookup goes on.
    ///
    /// For a shared-texture handle, on the link's queue, the hold has no holder: the
    /// reference goes to the worker's last resort, released once the worker has left the queue
    /// ([`crate::rt::pool::drain_idle`]). For an exec's invalidation, on the FIFO thread, the
    /// FIFO command's holder takes it. Both run it under a handler of their own.
    fn kernel_task(&self, hold: Hold<'_>) -> RaiseResult<Option<KernelTaskRef<P>>> {
        match self.device.upgrade() {
            Some(d) => {
                let k = d.get_kernel_task(OnFault::Return)?;
                hold.keep(Arc::clone(&k));
                Ok(Some(k))
            }
            None => Ok(None),
        }
    }

    /// Passes `reply` a new handle for the kernel task's shared texture `id`, or nil when the
    /// device is gone. A raise on the way is logged and `reply` gets nil.
    pub(crate) fn shared_texture_handle(
        &self,
        hold: Hold<'_>,
        id: u32,
        reply: &Reply<Option<P::SharedTextureHandle>>,
    ) {
        let r = (|| -> RaiseResult<_> {
            let kernel = self.kernel_task(hold)?;
            let handle = match &kernel {
                Some(k) => k.new_shared_texture_handle_for_id(hold, id)?,
                None => None,
            };
            // The reply takes the +1.
            reply.call(handle);
            Ok(())
        })();
        if let Err(e) = r {
            log::warn!(
                "no shared texture handle for backing {id}: {}",
                exception(&e)
            );
            reply.call(None);
            hold.land(e);
        }
    }

    /// Invalidates, for the guest, the kernel task's shared-texture backings an exec's reply
    /// listed (on the FIFO thread, after the exec). None listed asks nothing of the device; a
    /// raise is logged.
    pub(crate) fn invalidate_shared_texture_backings(&self, hold: Hold<'_>, backings: &[u32]) {
        let count = backings.len().low_u32();
        let r = (|| -> RaiseResult<_> {
            if count != 0
                && let Some(k) = self.kernel_task(hold)?
                && let Some(records) = Records::new(backings.as_bytes(), count)
            {
                k.invalidate_guest_for_shared_texture_backing(hold, records)?;
            }
            Ok(())
        })();
        if let Err(e) = r {
            log::warn!(
                "shared texture backings not invalidated for the guest: {}",
                exception(&e)
            );
            hold.land(e);
        }
    }

    /// Whether `mapping_id` has an entry, and whether it holds a surface.
    #[cfg(test)]
    pub(crate) fn kept(&self, mapping_id: u64) -> Option<bool> {
        self.surfaces.lock().get(&mapping_id).map(Option::is_some)
    }

    /// The number of kept entries.
    #[cfg(test)]
    pub(crate) fn kept_count(&self) -> usize {
        self.surfaces.lock().len()
    }

    /// Keeps `surface` under `mapping_id`, as a retain would.
    #[cfg(test)]
    pub(crate) fn keep(&self, mapping_id: u64, surface: P::Surface) {
        self.surfaces.lock().insert(mapping_id, Some(surface));
    }
}
