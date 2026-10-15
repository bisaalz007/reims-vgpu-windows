//! The command ring: the doorbell's handler runs every command from the read counter up to the
//! write counter, each under the queue state's lock from its ring read to its status store, then
//! bumps the read counter and raises interrupt 0 outside every lock.

// Guest data flows through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use std::sync::atomic::Ordering;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::iosurface::DeathKind;
use crate::iosurface::device::HostDeviceCore;
use crate::num;
use crate::platform::Platform;

/// One command's size in the ring.
pub(super) const RING_COMMAND_SIZE: u64 = 16;

/// A ring command (16 bytes).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct IoSurfaceDeviceRingCommand {
    /// 1 map, 2 unmap, 3 test; anything else is unsupported.
    pub command: u32,
    /// The surface's mapping id.
    pub mapping_id: u32,
    /// The command's [`IoReturn`], written back.
    pub status: i32,
    /// Never read.
    pub field_0c: u32,
}

const _: () = assert!(num::const_u64(size_of::<IoSurfaceDeviceRingCommand>()) == RING_COMMAND_SIZE);

/// What a command does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Command {
    /// Makes the surface a descriptor describes.
    Map,
    /// Releases it, when nothing else holds it.
    Unmap,
    /// Writes a test pattern into it.
    Test,
}

impl TryFrom<u32> for Command {
    type Error = u32;

    fn try_from(command: u32) -> Result<Command, u32> {
        match command {
            1 => Ok(Command::Map),
            2 => Ok(Command::Unmap),
            3 => Ok(Command::Test),
            other => Err(other),
        }
    }
}

/// A command's status, as the `IOReturn` stored back into the ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum IoReturn {
    /// `kIOReturnSuccess`.
    Success,
    /// `kIOReturnNoMemory`.
    NoMemory,
    /// `kIOReturnBadArgument`.
    BadArgument,
    /// `kIOReturnUnsupported`.
    Unsupported,
    /// `kIOReturnBusy`.
    Busy,
}

impl IoReturn {
    /// The `IOReturn` code.
    pub(super) const fn code(self) -> i32 {
        match self {
            IoReturn::Success => 0,
            IoReturn::NoMemory => 0xe000_02bd_u32.cast_signed(),
            IoReturn::BadArgument => 0xe000_02c2_u32.cast_signed(),
            IoReturn::Unsupported => 0xe000_02c7_u32.cast_signed(),
            IoReturn::Busy => 0xe000_02d5_u32.cast_signed(),
        }
    }
}

impl<P: Platform> HostDeviceCore<P> {
    /// Runs the ring from the read counter to the write counter (both re-read at each command,
    /// free-running `u64`s), storing each command's status and raising interrupt 0 per command.
    /// The state lock is not held across the host's block. The teardown does not wait for a
    /// command: once it has unmapped the ring, the command's status store or the next ring read
    /// faults; a ring a new size leaked stays readable.
    pub(super) fn run_ring(&self) {
        // A doorbell with no ring or no table runs nothing.
        if self.regs.ring_base.load(Ordering::SeqCst) == 0
            || self.regs.ring_size.load(Ordering::SeqCst) == 0
        {
            return;
        }
        {
            let st = self.state.lock();
            if st.ring.is_none() || st.table.is_none() {
                return;
            }
        }
        while self.regs.ring_rd_counter.load(Ordering::SeqCst)
            < self.regs.ring_wr_counter.load(Ordering::SeqCst)
        {
            let ring_size = self.regs.ring_size.load(Ordering::SeqCst);
            let ring_rd_counter = self.regs.ring_rd_counter.load(Ordering::SeqCst);
            let (command, mapping_id, status) = {
                let mut st = self.state.lock();
                let slot = st
                    .ring
                    .unwrap_or_default()
                    .offset(num::rem0(ring_rd_counter, ring_size).wrapping_mul(RING_COMMAND_SIZE));
                // Both words in one load pair: inside a 16-byte granule, as a slot of the
                // page-aligned ring is, it is single-copy atomic under FEAT_LSE2, so a guest's
                // 64-bit rewrite of the slot is seen whole or not at all, never one word of each.
                let (command, mapping_id) =
                    self.mapped_access(slot, 8, DeathKind::RingRead, || {
                        slot.read_u32_pair(0, DeathKind::RingRead)
                    });
                let status = match Command::try_from(command) {
                    Ok(Command::Map) => self.map(&mut st, mapping_id),
                    Ok(Command::Unmap) => self.unmap(&mut st, mapping_id),
                    Ok(Command::Test) => self.test_pattern(&st, mapping_id),
                    Err(command) => {
                        log::warn!("unsupported command {command} for surface {mapping_id}");
                        IoReturn::Unsupported
                    }
                };
                let at = slot.offset(8);
                self.mapped_access(at, 4, DeathKind::StatusStore, || {
                    at.write_u32(0, status.code().cast_unsigned(), DeathKind::StatusStore);
                });
                (command, mapping_id, status)
            };
            // Each failure's cause has its own line.
            if status != IoReturn::Success {
                log::debug!(
                    "command {command} for surface {mapping_id} answered {:#x}",
                    status.code()
                );
            }
            self.regs.ring_rd_counter.fetch_add(1, Ordering::SeqCst);
            // Outside the lock: the host's block may release the device, whose teardown then
            // runs here.
            self.raise_interrupt();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    use crate::iosurface::regs::{
        REG_DESCRIPTOR_BASE, REG_RING_BASE, REG_RING_RD_COUNTER, REG_RING_SIZE,
    };
    use crate::iosurface::testing::*;
    use crate::num::ToUsize;
    use crate::testkit::capture::expect_death;
    use crate::testkit::logs;

    /// Commands run in ring order, each with its status and interrupt 0; an unsupported command
    /// is a warning, and each failure's status is logged in hex; the counters wrap the ring by its
    /// size.
    #[test]
    fn runs_the_ring() {
        let logs = logs::capture();
        let r = rig();
        r.setup(true);
        r.root(1);
        r.put_v2(1, &v2(1, ROOT));
        r.command(0, 1, 1);
        r.command(1, 3, 1);
        r.command(2, 2, 1);
        r.command(3, 9, 1);
        r.command(4, 1, 0);
        r.doorbell(5);
        assert_eq!(
            (0..5).map(|s| r.status(s)).collect::<Vec<_>>(),
            [
                IoReturn::Success,
                IoReturn::Success,
                IoReturn::Success,
                IoReturn::Unsupported,
                IoReturn::BadArgument,
            ]
            .map(IoReturn::code)
        );
        assert_eq!(r.dev.mmio_read(REG_RING_RD_COUNTER), 5);
        assert_eq!(*r.irqs.lock(), vec![0; 5]);
        assert!(r.surface(1).is_none());
        assert_eq!(
            lines_from(&logs, module_path!()),
            [
                (
                    log::Level::Warn,
                    "unsupported command 9 for surface 1".to_owned()
                ),
                (
                    log::Level::Debug,
                    "command 9 for surface 1 answered 0xe00002c7".to_owned()
                ),
                (
                    log::Level::Debug,
                    "command 1 for surface 0 answered 0xe00002c2".to_owned()
                ),
            ]
        );
        // Wrap: counters 1024 and 1025 use slots 0 and 1.
        r.dev.state.lock().ring = None;
        r.dev.mmio_write(REG_RING_RD_COUNTER, 1024);
        r.dev.mmio_write(REG_RING_BASE, RING);
        r.dev.mmio_write(REG_RING_SIZE, 1024);
        r.command(0, 1, 1);
        r.command(1, 9, 2);
        r.doorbell(1026);
        assert_eq!(
            (r.status(0), r.status(1)),
            (IoReturn::Success.code(), IoReturn::Unsupported.code())
        );
        assert_eq!(r.dev.mmio_read(REG_RING_RD_COUNTER), 1026);
        assert!(r.surface(1).is_some());
    }

    /// No ring, no table or a zero register: nothing runs.
    #[test]
    fn needs_ring_and_table() {
        let r = rig();
        r.dev.mmio_write(REG_RING_BASE, RING);
        r.dev.mmio_write(REG_RING_SIZE, 1024);
        r.command(0, 9, 1);
        r.doorbell(1);
        assert_eq!(r.dev.mmio_read(REG_RING_RD_COUNTER), 0);
        assert_eq!(r.status(0), 0x5555);
        r.dev.mmio_write(REG_DESCRIPTOR_BASE, TABLE);
        r.dev.mmio_write(REG_RING_BASE, 0);
        r.doorbell(1);
        assert_eq!(r.dev.mmio_read(REG_RING_RD_COUNTER), 0);
        assert!(r.irqs.lock().is_empty());
        r.dev.run_ring();
        assert_eq!(r.dev.mmio_read(REG_RING_RD_COUNTER), 0);
    }

    /// A missing interrupt block is called anyway: the host dies after the first command.
    #[test]
    fn nil_interrupt_block_dies() {
        expect_death(DeathKind::NoInterrupt.site(), || {
            let r = rig_with(false);
            r.setup(true);
            r.command(0, 9, 1);
            r.doorbell(1);
        });
    }

    /// A guest vCPU rewrites the pending slot with one 64-bit store while the host drains the
    /// ring. The command and mapping id are read with one load pair: the ring is a mapping's
    /// page-aligned address and a slot is 16 bytes, so the pair sits in one 16-byte granule and
    /// FEAT_LSE2 makes it single-copy atomic. The command run, logged and answered is always one
    /// the guest wrote, never one word of each. Here the guest flips slot 0 between `{4, 7}` and
    /// `{5, 9}` (both unsupported, so each run logs the pair it read) until the host has run both.
    #[test]
    fn reads_a_command_in_one_access() {
        const A: &str = "unsupported command 4 for surface 7";
        const B: &str = "unsupported command 5 for surface 9";
        let logs = logs::capture();
        // Each run's warning: the pair it read.
        let mut pairs: Vec<String> = Vec::new();
        let take = |pairs: &mut Vec<String>| {
            pairs.extend(
                lines_from(&logs, module_path!())
                    .into_iter()
                    .filter(|l| l.0 == log::Level::Warn)
                    .map(|l| l.1),
            );
        };
        let r = rig();
        r.setup(true);
        let ring = r.dev.state.lock().ring.unwrap();
        // `{command, mapping_id}` as one little-endian 64-bit guest store.
        let a = (7u64 << 32) | 4;
        let b = (9u64 << 32) | 5;
        let site = DeathKind::RingRead.site();
        ring.write_u64(0, a, site);
        let (started, stop) = (AtomicBool::new(false), AtomicBool::new(false));
        let (mut runs, mut next_check, mut saw_a, mut saw_b) = (0u32, 20_000u32, false, false);
        std::thread::scope(|sc| {
            sc.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    ring.write_u64(0, b, site);
                    ring.write_u64(0, a, site);
                    started.store(true, Ordering::Relaxed);
                }
            });
            while !started.load(Ordering::Relaxed) {
                std::thread::yield_now();
            }
            // Until both commands have run, checking the log at doubling counts.
            while !(saw_a && saw_b) {
                assert!(runs < 400_000, "the guest's stores never interleaved");
                r.dev.regs.ring_rd_counter.store(0, Ordering::SeqCst);
                r.dev.regs.ring_wr_counter.store(1, Ordering::SeqCst);
                r.dev.run_ring();
                runs += 1;
                if runs == next_check {
                    next_check *= 2;
                    take(&mut pairs);
                    saw_a = pairs.iter().any(|l| l == A);
                    saw_b = pairs.iter().any(|l| l == B);
                }
            }
            stop.store(true, Ordering::Relaxed);
        });
        take(&mut pairs);
        assert_eq!(pairs.len(), runs.to_usize());
        let torn: Vec<_> = pairs.iter().filter(|p| *p != A && *p != B).collect();
        assert!(
            torn.is_empty(),
            "{} torn commands in {runs}, first {:?}",
            torn.len(),
            torn.first()
        );
        assert_eq!(r.dev.mmio_read(REG_RING_RD_COUNTER), 1);
        assert_eq!(r.irqs.lock().len(), runs.to_usize());
    }
}
