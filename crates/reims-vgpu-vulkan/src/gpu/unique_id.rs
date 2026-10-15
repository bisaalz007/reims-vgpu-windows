//! `uniqueIdentifier` (SPI) of pipeline states: the runtime has none, and the guest reads it
//! (the info command 460), so this platform hands one out itself.
//!
//! Metal's identifiers of states made without `supportIndirectCommandBuffers` differ from run to
//! run, so what the guest can rely on, and what the suite locks as relations (`info_replies`' 460
//! records, of such states), is how they relate: one pipeline answers one nonzero value every time;
//! a render and a compute pipeline answer different ones; and a second pipeline made from an
//! identical record answers a new one. So every creation takes the next value of one process-wide
//! counter, shared by every pipeline kind, even where the runtime answers a state it made before
//! for an equal request; the handle keeps it, so every reference to that handle answers it.
//!
//! **By resource ID.** An indirect command's record names its pipeline state by the state's
//! resource ID, and a record the GPU wrote (a kernel's `set_compute_pipeline_state` or
//! `set_render_pipeline_state` of a state it read from an argument buffer, or a blit's copy) names
//! one no setter gave the command. Metal's `getPipelineStateUniqueIdentifier` answers that
//! state's own identifier (measured on the M3 Max, macOS 26.7: of a compute and of a render
//! command a kernel encoded, and of the second of two states made from one descriptor), and still
//! answers it once the state was released, though a state made later took its ID (measured). So
//! each identifier handed out for a state of a nonzero resource ID, one made with
//! `supportIndirectCommandBuffers`, is kept by that ID and its device with the points of [`now`]'s
//! order it was made and released at ([`next_of`], [`released`], [`holders`]): a state holds its
//! ID from its creation to its release. The runtime gives a state's ID back only once the state
//! goes, so states its plain allocation numbered hold one ID one after another; a state placed at
//! the ID of one still live holds it beside that one. The [`HISTORY`] states made last with each
//! ID are kept. Which of them a record names is decided by when it can have been written
//! ([`crate::gpu::icb`]).
//!
//! **Of a state with indirect command buffer support.** Metal's identifier of a state made with
//! `supportIndirectCommandBuffers` is an address of its own allocation of such states,
//! `0x1000000000` and up, the same in every run that makes and releases the same states in the same
//! order (measured on the M3 Max, macOS 26.7: five runs of a sequence of twelve creations and
//! releases answered equal values), which it reuses once a state goes: a compute state made after
//! one was released answers the released one's identifier, the last released first, whatever
//! resource ID it takes (of two released, it took the first one's ID and the second one's
//! identifier), and a render state takes another kind of allocation. Its steps depend on the
//! functions (one kernel's states 0xc0 apart, the first of another 0x300 on, render states 0x200
//! apart), and its base on what the process made before and on how it was started (four states made
//! by a test binary its test started answered each 0x100 below the same four made by the binary
//! started from a shell). This platform answers a new identifier for every creation, and reproduces
//! none of that, so the identifier of every state made with `supportIndirectCommandBuffers` dies
//! where the guest reads it (info command 460, [`crate::gpu::resource`]). A state made without that
//! support has an identifier Metal allocates elsewhere, as any memory, which differs from run to
//! run, and the guest's read answers this platform's.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

use reims_vgpu_core::sync::Mutex;

/// The next identifier to hand out; 0 is never one.
static NEXT: AtomicU64 = AtomicU64::new(1);

/// A fresh identifier for a pipeline state being made: nonzero, and never handed out before in
/// this process.
pub fn next() -> u64 {
    // Relaxed: only the values' uniqueness matters, not their order against other memory.
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// The last point handed out of the order of what decides which state a record names: each
/// state's creation and each span in which the GPU may write records ([`now`]).
static ORDER: AtomicU64 = AtomicU64::new(0);

/// The next point of that order, later than every point handed out before, on any thread.
pub fn now() -> u64 {
    // One location's read-modify-writes take one total order, so a point taken after another
    // returned is larger.
    ORDER.fetch_add(1, Ordering::SeqCst) + 1
}

/// How many of the states made last with one resource ID on one device are kept ([`History`]).
pub const HISTORY: usize = 16;

/// A point of [`now`]'s order no point reaches: the release of a state still live.
pub const LIVE: u64 = u64::MAX;

/// A state made with a resource ID, which held the ID from `made` to `gone`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Holder {
    /// Its identifier.
    pub identifier: u64,
    /// The point of [`now`]'s order it was made at.
    pub made: u64,
    /// The point it was released at; [`LIVE`] while it lives.
    pub gone: u64,
}

/// The states made with one resource ID on one device, oldest first: the [`HISTORY`] made last,
/// those released first giving way to newer ones.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct History {
    /// The states kept.
    pub holders: VecDeque<Holder>,
    /// The point until which a state no longer kept may have held the ID: 0 where none was
    /// dropped, [`LIVE`] where one was dropped while it lived.
    pub lost_until: u64,
}

/// The states made with each nonzero resource ID, by the registry ID of the runtime's device
/// they were made on and that ID.
static OF_ID: Mutex<BTreeMap<(u64, u64), History>> = Mutex::new(BTreeMap::new());

/// The point of [`now`]'s order at which the last state of a nonzero resource ID was made, by
/// the registry ID of the runtime's device it was made on.
static LAST_MADE: Mutex<BTreeMap<u64, u64>> = Mutex::new(BTreeMap::new());

/// A fresh identifier ([`next`]) for a state the runtime just made on the device of registry ID
/// `device` with resource ID `id`, kept by that ID with the point it was made at ([`holders`]).
/// A state of ID 0, which no indirect command can name, is kept by none.
pub fn next_of(device: u64, id: u64) -> u64 {
    let identifier = next();
    if id == 0 {
        return identifier;
    }
    let mut of_id = OF_ID.lock();
    let history = of_id.entry((device, id)).or_default();
    // Taken under the lock, so each ID's states are kept in the order they were made.
    let made = now();
    history.holders.push_back(Holder {
        identifier,
        made,
        gone: LIVE,
    });
    LAST_MADE.lock().insert(device, made);
    if history.holders.len() > HISTORY {
        // The oldest released one, else the oldest.
        let at = (history.holders.iter())
            .position(|holder| holder.gone != LIVE)
            .unwrap_or(0);
        if let Some(dropped) = history.holders.remove(at) {
            history.lost_until = history.lost_until.max(dropped.gone);
        }
    }
    identifier
}

/// The states made on the device of registry ID `device` with resource ID `id`; `None` where
/// none was.
pub fn holders(device: u64, id: u64) -> Option<History> {
    OF_ID.lock().get(&(device, id)).cloned()
}

/// Notes that the state of identifier `identifier` made on the device of registry ID `device`
/// with resource ID `id` was released, now: it no longer holds the ID. One of ID 0 changes
/// nothing.
pub fn released(device: u64, id: u64, identifier: u64) {
    if id == 0 {
        return;
    }
    let gone = now();
    let mut of_id = OF_ID.lock();
    let holder = (of_id.get_mut(&(device, id))).and_then(|history| {
        history
            .holders
            .iter_mut()
            .find(|h| h.identifier == identifier)
    });
    if let Some(holder) = holder {
        holder.gone = gone;
    }
}

/// Whether a state kept by its nonzero resource ID on the device of registry ID `device` was both
/// made and released from point `from` up to point `to` of [`now`]'s order.
pub fn held_within(device: u64, from: u64, to: u64) -> bool {
    let of_id = OF_ID.lock();
    let of_device = of_id.range((device, 0)..=(device, u64::MAX));
    of_device
        .flat_map(|(_, history)| &history.holders)
        .any(|holder| holder.made >= from && holder.gone != LIVE && holder.gone <= to)
}

/// Whether a state of a nonzero resource ID was made on the device of registry ID `device` after
/// point `point` of [`now`]'s order.
pub fn made_since(device: u64, point: u64) -> bool {
    LAST_MADE
        .lock()
        .get(&device)
        .is_some_and(|&made| made > point)
}

/// What these keep is process-wide, so tests over a real device's states need a process each, as
/// `cargo nextest` gives them; these use registry IDs no device answers.
#[cfg(test)]
mod tests {
    use super::*;

    /// Every identifier is nonzero and new, across threads
    #[test]
    fn identifiers_are_nonzero_and_never_reused() {
        let mut seen: Vec<u64> = std::thread::scope(|s| {
            let workers: Vec<_> = (0..4)
                .map(|_| s.spawn(|| (0..256).map(|_| next()).collect::<Vec<_>>()))
                .collect();
            workers
                .into_iter()
                .flat_map(|w| w.join().unwrap())
                .collect()
        });
        let count = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), count);
        assert!(seen.iter().all(|&id| id != 0));
    }

    /// The states made with a nonzero resource ID are kept by it, in the order they were made,
    /// with when each was released, on their own device only, the [`HISTORY`] made last, a
    /// released one giving way first; one of ID 0 is kept by none, and its release changes
    /// nothing
    #[test]
    fn states_are_kept_by_resource_id_in_the_order_they_were_made() {
        // A registry ID no device answers, so no other test's states meet these.
        let device = u64::MAX - 7;
        let before = now();
        let first = next_of(device, 3);
        let kept = holders(device, 3).unwrap();
        assert_eq!((kept.holders.len(), kept.lost_until), (1, 0));
        let holder = kept.holders[0];
        assert_eq!((holder.identifier, holder.gone), (first, LIVE));
        assert!(holder.made > before && holder.made < now());
        released(device, 0, first);
        assert_eq!(holders(device, 3).unwrap().holders[0].gone, LIVE);
        released(device, 3, first);
        let later = next_of(device, 3);
        assert_ne!(later, first);
        let kept = holders(device, 3).unwrap();
        let (a, b) = (kept.holders[0], kept.holders[1]);
        assert_eq!((a.identifier, b.identifier, b.gone), (first, later, LIVE));
        assert!(a.made < a.gone && a.gone < b.made);
        assert_eq!(holders(device - 1, 3), None);
        let unnamed = next_of(device, 0);
        assert_ne!(unnamed, 0);
        assert_eq!(holders(device, 0), None);
        // Past HISTORY, the released first state gives way, then the oldest live one.
        let made: Vec<_> = (1..HISTORY).map(|_| next_of(device, 3)).collect();
        let kept = holders(device, 3).unwrap();
        assert_eq!(kept.lost_until, a.gone);
        let identifiers: Vec<_> = kept.holders.iter().map(|h| h.identifier).collect();
        assert_eq!(identifiers[0], later);
        assert_eq!(identifiers[1..], made[..]);
        next_of(device, 3);
        assert_eq!(holders(device, 3).unwrap().lost_until, LIVE);
    }
}
