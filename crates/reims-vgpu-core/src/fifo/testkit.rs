//! Scaffolding for the FIFO's unit tests: a real device over a fake platform, guest memory for
//! rings and nodes.

#![allow(clippy::disallowed_methods)]

use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Weak};

use zerocopy::IntoBytes;

use crate::device::DeviceCore;
use crate::device::test_support::{Opts, TestDevice, device};
use crate::fifo::Fifo;
use crate::fifo::wire::{FifoCommand, StampWait};
use crate::guest::GuestPtr;
use crate::guest::records::GuestFeatures;
use crate::testkit::FakePlatform;
use crate::testkit::guest_ram::RawRegion;
use crate::testkit::host::FakeHost;

pub(crate) type P = FakePlatform;

/// A fake platform, the device the FIFOs signal (with stamp memory), and the device's
/// last-pending cell.
pub(crate) struct Kit {
    pub platform: Arc<P>,
    pub td: TestDevice,
    pub host: Arc<DeviceCore<P>>,
    pub last_pending: Arc<AtomicU32>,
}

impl Kit {
    pub(crate) fn new() -> Kit {
        let td = device(Opts::default()).with_stamp_memory().with_root_page();
        let host: Arc<DeviceCore<P>> = Arc::clone(&td.device);
        Kit {
            platform: td.platform.clone(),
            td,
            host,
            last_pending: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Guest RAM from physical page `page` on.
    pub(crate) fn guest_pages(&self, page: u64) -> GuestPages {
        GuestPages {
            platform: self.platform.clone(),
            host: self.td.host.clone(),
            pa: page * crate::guest::records::APV_PAGE_SIZE,
        }
    }

    pub(crate) fn weak(&self) -> Weak<DeviceCore<P>> {
        Arc::downgrade(&self.host)
    }

    /// Changes the device's features.
    pub(crate) fn features(&self, f: impl FnOnce(&mut GuestFeatures)) {
        let shared = self.host.features();
        let mut a = shared.load();
        f(&mut a);
        shared.store(&a);
    }

    /// A root FIFO over a fresh ring of `length` bytes starting at `start`.
    pub(crate) fn root(&self, length: u32, start: u32) -> (Arc<Fifo<P>>, Arc<RawRegion>) {
        let ram = RawRegion::new(u64::from(length.max(1))).unwrap();
        let f = self.root_at(length, GuestPtr::new(ram.addr()), start);
        (f, ram)
    }

    /// A root FIFO over `length` bytes at `base` (refused or not).
    pub(crate) fn root_at(&self, length: u32, base: GuestPtr, start: u32) -> Arc<Fifo<P>> {
        Fifo::init_root(
            &self.platform,
            self.weak(),
            length,
            base,
            start,
            self.last_pending.clone(),
        )
        .0
    }

    /// A child FIFO over `node`.
    pub(crate) fn child(&self, node: GuestPtr, restoring: bool) -> Option<Arc<Fifo<P>>> {
        Fifo::new_child_on(
            &self.platform,
            self.weak(),
            node,
            1,
            restoring,
            self.last_pending.clone(),
        )
    }

    /// `(stamp_index, value)` of every stamp the device signalled, by index: the stamp
    /// interrupt's bits and the stamp slots they name.
    pub(crate) fn signals(&self) -> Vec<(u32, u32)> {
        let bits = self.td.interrupt_stamp_bits();
        let ram = self.td.stamp_ram.as_ref().expect("stamp ram");
        (0..32)
            .filter(|i| bits & (1 << i) != 0)
            .map(|i| (i, ram.read_u32(u64::from(i) * 4).unwrap()))
            .collect()
    }

    /// How many FIFOs' faults reached the device (the fault interrupt's bits), once the FIFO
    /// queue has run the fault blocks queued so far.
    pub(crate) fn faults(&self) -> usize {
        self.td.device.on_fifo_queue(|_| ());
        self.td.interrupt_fault_bits().count_ones() as usize
    }
}

/// A command: header, barriers, payload; `length` and `barrier_count` override the header's.
pub(crate) fn cmd(
    id: u16,
    barriers: &[StampWait],
    stamp: u32,
    payload: &[u8],
    length: Option<u32>,
    barrier_count: Option<u16>,
) -> Vec<u8> {
    let total = 12 + 8 * barriers.len() + payload.len();
    let h = FifoCommand {
        command_id: id,
        barrier_count: barrier_count.unwrap_or(barriers.len() as u16),
        length: length.unwrap_or(total as u32),
        stamp_value: stamp,
    };
    let mut v = h.as_bytes().to_vec();
    for b in barriers {
        v.extend_from_slice(b.as_bytes());
    }
    v.extend_from_slice(payload);
    v
}

/// Polls `cond` for up to 10 s of real time.
pub(crate) fn wait_for(mut cond: impl FnMut() -> bool) -> bool {
    for _ in 0..10_000 {
        if cond() {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    cond()
}

/// A child node (0x14 bytes) and its FIFO's ring (guest pages from 0x20, listed in guest page
/// 7), which the device maps from guest RAM.
pub(crate) struct ChildMem {
    pub node: Arc<RawRegion>,
    pub ring: GuestPages,
}

impl ChildMem {
    /// `pages` ring pages; the node names stamp index `stamp_index` and list page 7.
    pub(crate) fn new(kit: &Kit, pages: u32, stamp_index: u16) -> ChildMem {
        let node = RawRegion::new(0x4000).unwrap();
        node.write(0xc, &stamp_index.to_le_bytes());
        node.write_u32(0x10, 7);
        let list = kit.guest_pages(7);
        for i in 0..pages {
            list.write_u32(u64::from(i) * 4, 0x20 + i);
        }
        ChildMem {
            node,
            ring: kit.guest_pages(0x20),
        }
    }

    pub(crate) fn node_ptr(&self) -> GuestPtr {
        GuestPtr::new(self.node.addr())
    }
}

/// Guest RAM from a physical page on; a write reaches the device's mappings of it.
pub(crate) struct GuestPages {
    platform: Arc<P>,
    host: Arc<FakeHost>,
    pa: u64,
}

impl GuestPages {
    pub(crate) fn write(&self, off: u64, src: &[u8]) -> bool {
        self.platform.fake_vm().sync();
        let ok = self.host.ram().write(self.pa + off, src);
        self.platform.fake_vm().sync();
        ok
    }

    pub(crate) fn write_u32(&self, off: u64, v: u32) -> bool {
        self.write(off, &v.to_le_bytes())
    }
}
