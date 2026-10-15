//! The machines composed on `FakePlatform`, with no seam fakes left.
//!
//! The only doubles are the platform's (`FakePlatform`: Metal, the VM, the launcher) and the
//! host's (`FakeHost`'s blocks over fake guest RAM). The fake GPU executes nothing: an exec is
//! observed as the calls it makes on the fake Metal, and completes when the test completes its
//! committed command buffers.

#![cfg(feature = "testkit")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_truncation
)] // A failure is the assertion

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use reims_vgpu_core::device::Device;
use reims_vgpu_core::device::config::DeviceConfig;
use reims_vgpu_core::host::HostBlocks;
use reims_vgpu_core::memory::descriptor::{GuestPhysicalRange, MemoryMapDescriptor};
use reims_vgpu_core::platform::types::NsRange;
use reims_vgpu_core::raise::death::DeathSite;
use reims_vgpu_core::rt::Hold;
use reims_vgpu_core::testkit::calllog::Arg;
use reims_vgpu_core::testkit::capture::DeathSlot;
use reims_vgpu_core::testkit::capture::with_slot;
use reims_vgpu_core::testkit::guest_ram::GuestRam;
use reims_vgpu_core::testkit::host::FakeHost;
use reims_vgpu_core::testkit::objects::CompleteOn;
use reims_vgpu_core::testkit::platform::FakePlatform;
use reims_vgpu_core::testkit::script::{Step, metal_crash_site};

type P = FakePlatform;

// ---------------------------------------------------------------------------------------------
// Guest memory layout
// ---------------------------------------------------------------------------------------------

/// The guest's page.
const PAGE: u64 = reims_vgpu_core::guest::records::APV_PAGE_SIZE;
/// Guest RAM: GPA 0x40000000, as the measured sequences place it (docs/testing.md 6.25).
const RAM_BASE: u64 = 0x4000_0000;
const RAM_PAGES: u64 = 128;
/// The first guest page number of RAM.
const BP: u32 = (RAM_BASE / PAGE) as u32;

const ROOT_PAGE: u32 = BP + 1;
const FIFO_PAGE: u32 = BP + 2;
/// One page.
const FIFO_LENGTH: u32 = PAGE as u32;
const FIFO_START: u32 = 0x400;
const RING: u32 = FIFO_LENGTH - FIFO_START;

/// Task `t`'s root page, page-table page and first data page (t in 1..=3).
fn task_root(t: u32) -> u32 {
    BP + 4 + 2 * (t - 1)
}
fn task_table(t: u32) -> u32 {
    task_root(t) + 1
}
fn task_data(t: u32) -> u32 {
    BP + 16 + TASK_PAGES as u32 * (t - 1)
}
const TASK_PAGES: u64 = 16;
const TASK_LEN: u64 = TASK_PAGES * PAGE;

/// Task VA layout: page 0 command buffer, page 1 records, page 2 object list, page 3 the buffer.
const VA_CMD: u64 = 0;
const VA_REC: u64 = PAGE;
const VA_OL: u64 = 2 * PAGE;
const VA_BUF_PAGE: u32 = 3;

// Registers (docs/testing.md 6.2).
const R_ENABLE: u64 = 0x1000;
const R_LENGTH: u64 = 0x1004;
const R_WRITTEN: u64 = 0x1008;
const R_READ: u64 = 0x100c;
const R_START: u64 = 0x1010;
const R_STAMP_IRQ: u64 = 0x1018;
const R_ROOT_PAGE: u64 = 0x101c;
const R_RESUME: u64 = 0x1024;
const R_FAULTED: u64 = 0x102c;
const R_BASE_PAGE: u64 = 0x1030;
const R_VERSION: u64 = 0x1034;
const R_FAULT_OFFSET: u64 = 0x103c;

// Opcodes (docs/testing.md 6.8).
const OP_NOP: u16 = 30;
const OP_DELETE_TASK: u16 = 32;
const OP_SET_OBJECT_LIST: u16 = 51;
const OP_DEFINE_TASK2: u16 = 56;
const OP_MAP_MEMORY2: u16 = 57;
const OP_EXEC_INDIRECT3: u16 = 67;
const OP_INVALID: u16 = 13;

/// The blit fill the exec runs (`FillBuffer`, 306): 0x20 bytes of 0xAB at buffer offset 0x10.
const FILL_AT: u64 = 0x10;
const FILL_LEN: u64 = 0x20;
const FILL_VALUE: u8 = 0xab;

// ---------------------------------------------------------------------------------------------
// The rig
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Setup {
    /// QEMU's eight blocks, no memory map.
    Blocks,
    /// A memory map and `raiseInterrupt` only.
    Map,
    /// A memory map and process isolation: user tasks are `RemoteTask`s over `ThreadLauncher`.
    Isolate,
}

struct Guest {
    setup: Setup,
    platform: Arc<P>,
    ram: GuestRam,
    host: Option<Arc<FakeHost>>,
    irqs: Arc<Mutex<Vec<u32>>>,
    device: Option<Device<P>>,
    slot: Arc<DeathSlot>,
    /// The root FIFO's free-running write count.
    written: u32,
    /// The next stamp value.
    stamp: u32,
}

fn map_descriptor(ram: &GuestRam) -> MemoryMapDescriptor {
    let mut d = MemoryMapDescriptor::new();
    for r in ram.ranges() {
        d.add_range(GuestPhysicalRange {
            physical_address: r.physical_address,
            physical_length: r.region.len(),
            host_address: r.host(),
        });
    }
    d
}

/// Polls `f` in real time (the device's threads are real threads) for up to about 10 s.
fn eventually(what: &str, mut f: impl FnMut() -> bool) {
    for _ in 0..10_000 {
        if f() {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("timed out waiting for {what}");
}

/// [`eventually`], printing `dump` before it fails.
fn eventually_or(what: &str, dump: impl Fn() -> String, mut f: impl FnMut() -> bool) {
    for _ in 0..10_000 {
        if f() {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("timed out waiting for {what}\n{}", dump());
}

impl Guest {
    fn new(setup: Setup) -> Guest {
        let ram = GuestRam::with_range(RAM_BASE, RAM_PAGES * PAGE).expect("ram");
        // Isolate runs `ThreadLauncher`'s default wiring: one `GpuTask` per task on threads of
        // its own, over the same platform and the same guest RAM.
        let platform = Arc::new(FakePlatform::new());
        let irqs = Arc::new(Mutex::new(Vec::new()));
        let slot = DeathSlot::new();
        let (device, host) = with_slot(Some(slot.clone()), || {
            Guest::make_device(setup, &platform, &ram, &irqs)
        });
        Guest {
            setup,
            platform,
            ram,
            host,
            irqs,
            device: Some(device),
            slot,
            written: 0,
            stamp: 0,
        }
    }

    fn make_device(
        setup: Setup,
        platform: &Arc<P>,
        ram: &GuestRam,
        irqs: &Arc<Mutex<Vec<u32>>>,
    ) -> (Device<P>, Option<Arc<FakeHost>>) {
        let mut c = DeviceConfig::<P>::new(Some(platform.device()));
        let host = match setup {
            Setup::Blocks => {
                let host = FakeHost::new(platform, ram.clone());
                c.blocks = host.blocks();
                Some(host)
            }
            Setup::Map | Setup::Isolate => {
                platform.fake_vm().adopt(ram);
                let i = irqs.clone();
                c.blocks = HostBlocks {
                    raise_interrupt: Some(Arc::new(move |v| i.lock().push(v))),
                    ..HostBlocks::default()
                };
                c.memory_map_descriptor = Some(Arc::new(map_descriptor(ram)));
                c.enable_process_isolation = setup == Setup::Isolate;
                None
            }
        };
        let device = Device::new(platform.clone(), c).expect("device");
        (device, host)
    }

    fn dev(&self) -> &Device<P> {
        self.device.as_ref().expect("device")
    }

    fn in_slot<R>(&self, f: impl FnOnce() -> R) -> R {
        with_slot(Some(self.slot.clone()), f)
    }

    fn mmio_write(&self, offset: u64, value: u32) {
        self.in_slot(|| self.dev().mmio_write(Hold::NONE, offset, value));
    }

    fn mmio_read(&self, offset: u64) -> u32 {
        self.in_slot(|| self.dev().mmio_read(offset))
    }

    fn interrupts(&self) -> Vec<u32> {
        match &self.host {
            Some(h) => h.interrupts(),
            None => self.irqs.lock().clone(),
        }
    }

    // -- guest RAM --------------------------------------------------------------------------

    /// Brings every alias of guest RAM up to date (the fake VM copies at a remap and syncs at
    /// sync points).
    fn sync(&self) {
        self.platform.fake_vm().sync();
    }

    fn write(&self, gpa: u64, bytes: &[u8]) {
        self.sync();
        assert!(self.ram.write(gpa, bytes), "guest write at {gpa:#x}");
        self.sync();
    }

    fn read_u32(&self, gpa: u64) -> u32 {
        self.sync();
        self.ram.read_u32(gpa).expect("guest read")
    }

    fn read_u64(&self, gpa: u64) -> u64 {
        self.sync();
        self.ram.read_u64(gpa).expect("guest read")
    }

    fn page(p: u32) -> u64 {
        u64::from(p) * PAGE
    }

    // -- bring-up ---------------------------------------------------------------------------

    /// Version 245, the root page, and the root FIFO (docs/testing.md 6.25 (a)).
    fn bring_up(&mut self) {
        self.mmio_write(R_VERSION, 245);
        assert_eq!(self.mmio_read(R_VERSION), 245);
        self.mmio_write(R_ROOT_PAGE, ROOT_PAGE);
        assert_eq!(self.mmio_read(R_ROOT_PAGE), ROOT_PAGE);
        self.mmio_write(R_BASE_PAGE, FIFO_PAGE);
        self.mmio_write(R_LENGTH, FIFO_LENGTH);
        self.mmio_write(R_START, FIFO_START);
        self.mmio_write(R_ENABLE, 1);
        assert_eq!(self.mmio_read(R_ENABLE), 1, "the root FIFO is enabled");
        assert_eq!(self.mmio_read(R_FAULT_OFFSET), u32::MAX, "healthy");
        self.written = 0;
    }

    // -- the root FIFO ----------------------------------------------------------------------

    /// Root stamp slot 0 (the stamp memory is the FIFO's base).
    fn published_stamp(&self) -> u32 {
        self.read_u32(Guest::page(FIFO_PAGE))
    }

    /// Writes one command into the ring (not yet rung). Returns (its ring offset, its stamp).
    fn push(&mut self, op: u16, payload: &[u8]) -> (u32, u32) {
        self.stamp += 1;
        let len = 12 + payload.len() as u32;
        let mut c = Vec::new();
        c.extend_from_slice(&op.to_le_bytes());
        c.extend_from_slice(&0u16.to_le_bytes());
        c.extend_from_slice(&len.to_le_bytes());
        c.extend_from_slice(&self.stamp.to_le_bytes());
        c.extend_from_slice(payload);
        let offset = self.written % RING;
        let ring = Guest::page(FIFO_PAGE) + u64::from(FIFO_START);
        for (i, b) in c.iter().enumerate() {
            let at = (offset + i as u32) % RING;
            self.write(ring + u64::from(at), &[*b]);
        }
        self.written = self.written.wrapping_add(len);
        (offset, self.stamp)
    }

    fn ring(&self) {
        self.mmio_write(R_WRITTEN, self.written);
    }

    /// Completes the committed command buffers of every process until the root stamp reaches
    /// `stamp`.
    fn wait_stamp(&self, stamp: u32) {
        eventually_or(
            &format!("stamp {stamp}"),
            || self.dump(),
            || {
                self.complete_gpu();
                self.no_host_death();
                self.published_stamp() >= stamp
            },
        );
    }

    /// The fake GPU finishes what was committed, the host's and every service's.
    fn complete_gpu(&self) {
        let _ = self.platform.world().complete_all(CompleteOn::Thread);
    }

    /// Waits for the root FIFO's fault (0x102c) and returns the fault offset (0x103c).
    fn wait_fault(&self) -> u32 {
        eventually_or(
            "a fault",
            || self.dump(),
            || {
                self.complete_gpu();
                self.no_host_death();
                self.mmio_read(R_FAULTED) == 1
            },
        );
        self.mmio_read(R_FAULT_OFFSET)
    }

    fn resume(&self) {
        self.mmio_write(R_RESUME, 0);
    }

    /// What a failure shows: the host's calls, and the services' deaths.
    fn dump(&self) -> String {
        let mut out = format!(
            "setup {:?}, stamp {}, faulted {}, fault offset {:#x}\n",
            self.setup,
            self.published_stamp(),
            self.mmio_read(R_FAULTED),
            self.mmio_read(R_FAULT_OFFSET)
        );
        out.push_str(&format!("  calls: {:?}\n", self.platform.call_log().sels()));
        for (i, l) in self
            .platform
            .thread_launcher()
            .launched()
            .iter()
            .enumerate()
        {
            out.push_str(&format!(
                "service {i}: death {:?}, ended {}\n",
                l.death(),
                l.ended()
            ));
        }
        out
    }

    fn no_host_death(&self) {
        if let Some(d) = self.slot.death() {
            panic!("the host died: {d:?}");
        }
    }

    /// Runs `op` and waits for its stamp.
    fn run(&mut self, op: u16, payload: &[u8]) -> u32 {
        let (_, s) = self.push(op, payload);
        self.ring();
        self.wait_stamp(s);
        s
    }

    // -- tasks ------------------------------------------------------------------------------

    /// Task `t`'s root (depth 1) and page table: every page writable.
    fn lay_out_task(&self, t: u32) {
        let mut root = Vec::new();
        root.extend_from_slice(&task_table(t).to_le_bytes());
        root.extend_from_slice(&1u32.to_le_bytes());
        root.resize(40, 0);
        self.write(Guest::page(task_root(t)), &root);
        let mut pt = Vec::new();
        for i in 0..TASK_PAGES as u32 {
            pt.extend_from_slice(&((task_data(t) + i) | 1 << 31).to_le_bytes());
        }
        self.write(Guest::page(task_table(t)), &pt);
    }

    /// Task `t`'s VA `va` as a GPA.
    fn task_gpa(t: u32, va: u64) -> u64 {
        Guest::page(task_data(t)) + va
    }

    fn define_payload(t: u32) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(&(t << 1).to_le_bytes());
        p.extend_from_slice(&TASK_LEN.to_le_bytes());
        p.extend_from_slice(&task_root(t).to_le_bytes());
        p
    }

    fn map_payload(t: u32) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(&t.to_le_bytes());
        p.extend_from_slice(&0u64.to_le_bytes());
        p.extend_from_slice(&TASK_LEN.to_le_bytes());
        p
    }

    /// The object list (entry 0 invalid, entry 1 a shared buffer over VA page 3), its record,
    /// and a blit segment filling the buffer, all in task `t`'s memory.
    fn lay_out_objects(&self, t: u32) {
        // Record: `{u64 length, u32 task VA page, u8 flags}`, 16 bytes.
        let mut rec = Vec::new();
        rec.extend_from_slice(&PAGE.to_le_bytes());
        rec.extend_from_slice(&VA_BUF_PAGE.to_le_bytes());
        rec.extend_from_slice(&[0, 0, 0, 0]);
        self.write(Guest::task_gpa(t, VA_REC), &rec);
        // Object list: `{u32 type | length << 8, u64 VA}` per entry.
        let mut ol = Vec::new();
        ol.extend_from_slice(&0u32.to_le_bytes());
        ol.extend_from_slice(&0u64.to_le_bytes());
        ol.extend_from_slice(&(1u32 | 16 << 8).to_le_bytes());
        ol.extend_from_slice(&VA_REC.to_le_bytes());
        self.write(Guest::task_gpa(t, VA_OL), &ol);
        // Command buffer: one blit segment with one FillBuffer (306).
        let mut seg = Vec::new();
        seg.extend_from_slice(&0x28u32.to_le_bytes());
        seg.extend_from_slice(&[2, 0, 0, 0]);
        seg.extend_from_slice(&306u32.to_le_bytes());
        seg.extend_from_slice(&(8u32 + 0x18).to_le_bytes());
        seg.extend_from_slice(&1u32.to_le_bytes());
        seg.extend_from_slice(&FILL_AT.to_le_bytes());
        seg.extend_from_slice(&FILL_LEN.to_le_bytes());
        seg.extend_from_slice(&[FILL_VALUE, 0, 0, 0]);
        assert_eq!(seg.len(), 0x28);
        self.write(Guest::task_gpa(t, VA_CMD), &seg);
    }

    fn object_list_payload(t: u32) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(&t.to_le_bytes());
        p.extend_from_slice(&((VA_OL / PAGE) as u32).to_le_bytes());
        p.extend_from_slice(&24u32.to_le_bytes());
        p
    }

    /// ExecIndirect3: the buffer (ref 1) listed with flags 0, one command-buffer range.
    fn exec_payload(t: u32) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(&t.to_le_bytes());
        p.extend_from_slice(&1u32.to_le_bytes());
        p.extend_from_slice(&1u32.to_le_bytes());
        p.extend_from_slice(&1u32.to_le_bytes());
        p.extend_from_slice(&0u32.to_le_bytes());
        p.extend_from_slice(&VA_CMD.to_le_bytes());
        p.extend_from_slice(&0x28u64.to_le_bytes());
        p
    }

    /// Defines, maps and lists task `t`'s objects, checking the host's writes.
    fn define_task(&mut self, t: u32) {
        self.lay_out_task(t);
        self.lay_out_objects(t);
        self.run(OP_DEFINE_TASK2, &Guest::define_payload(t));
        assert_eq!(
            self.read_u32(Guest::page(ROOT_PAGE) + 4 * u64::from(t)),
            task_root(t),
            "root word[{t}] names the task root"
        );
        assert_eq!(
            self.read_u64(Guest::page(task_root(t)) + 0x14),
            TASK_LEN,
            "the host wrote the task length"
        );
        self.run(OP_MAP_MEMORY2, &Guest::map_payload(t));
        self.run(OP_SET_OBJECT_LIST, &Guest::object_list_payload(t));
    }

    /// Every `fillBuffer:range:value:` Metal received, from the host and from every service:
    /// the range and the value.
    fn fills(&self) -> Vec<(Arg, u8)> {
        let mut out = Vec::new();
        for c in self.platform.call_log().calls_to("fillBuffer:range:value:") {
            let value = match c.args[2] {
                Arg::U8(v) => v,
                _ => panic!("fill value"),
            };
            out.push((c.args[1].clone(), value));
        }
        out
    }

    /// What one exec's `FillBuffer` sends Metal.
    fn expected_fill() -> (Arg, u8) {
        let range = NsRange {
            location: FILL_AT,
            length: FILL_LEN,
        };
        (Arg::val(&range), FILL_VALUE)
    }

    fn exec(&mut self, t: u32) -> u32 {
        self.run(OP_EXEC_INDIRECT3, &Guest::exec_payload(t))
    }

    fn launched_deaths(&self) -> Vec<Option<DeathSite>> {
        self.platform
            .thread_launcher()
            .launched()
            .iter()
            .map(|l| l.death())
            .collect()
    }

    /// Drops the device, as the host's last release does.
    fn release(&mut self) {
        let d = self.device.take();
        self.in_slot(|| drop(d));
    }
}

impl Drop for Guest {
    fn drop(&mut self) {
        if std::thread::panicking() {
            // A failed test leaves its device unreleased: its teardown may wait for the work
            // that failed to finish.
            std::mem::forget(self.device.take());
        } else {
            self.release();
            self.no_host_death();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------------------------

/// Bring-up and one NOP: the stamp, the read count and the stamp interrupt (6.25 (a)).
fn bring_up_and_nop(setup: Setup) {
    let mut g = Guest::new(setup);
    g.bring_up();
    let s = g.run(OP_NOP, &[]);
    assert_eq!(s, 1);
    assert_eq!(g.published_stamp(), 1);
    assert_eq!(g.mmio_read(R_READ), 12);
    eventually("the stamp interrupt", || !g.interrupts().is_empty());
    assert_eq!(g.mmio_read(R_STAMP_IRQ), 1, "stamp bit 0, read to clear");
    assert_eq!(g.mmio_read(R_STAMP_IRQ), 0);
    // Three NOPs, one doorbell.
    for _ in 0..3 {
        let _ = g.push(OP_NOP, &[]);
    }
    g.ring();
    g.wait_stamp(4);
    assert_eq!(g.mmio_read(R_READ), 48);
}

/// A task defined, mapped and given objects; an exec's blit reaches Metal (6.25 (b), (c)).
fn task_and_exec(setup: Setup) {
    let mut g = Guest::new(setup);
    g.bring_up();
    g.define_task(1);
    // MapMemory2 reaches the host's blocks only in the blocks setup (docs/testing.md 6.10).
    let host_maps = g.platform.call_log().count("mapMemory");
    if setup == Setup::Blocks {
        assert!(host_maps > 0, "mapMemory through the host's block");
        assert!(!g.host.as_ref().expect("host").tasks().is_empty());
    } else {
        assert_eq!(host_maps, 0);
    }
    g.exec(1);
    let fills = g.fills();
    assert_eq!(fills, vec![Guest::expected_fill()]);
    assert_eq!(g.mmio_read(R_FAULTED), 0);
    // Delete: the root word goes.
    let mut p = Vec::new();
    p.extend_from_slice(&1u32.to_le_bytes());
    g.run(OP_DELETE_TASK, &p);
    assert_eq!(g.read_u32(Guest::page(ROOT_PAGE) + 4), 0);
}

/// An invalid opcode between two NOPs faults and holds the stamp; 0x1024 resumes (6.6).
fn fault_and_resume(setup: Setup) {
    let mut g = Guest::new(setup);
    g.bring_up();
    let _ = g.push(OP_NOP, &[]);
    let (bad, _) = g.push(OP_INVALID, &[]);
    let _ = g.push(OP_NOP, &[]);
    g.ring();
    let offset = g.wait_fault();
    assert_eq!(offset, bad);
    assert_eq!(offset, 0xc);
    assert_eq!(g.published_stamp(), 1, "held at the fault");
    g.resume();
    g.wait_stamp(3);
    assert_eq!(g.mmio_read(R_FAULT_OFFSET), u32::MAX);
    assert_eq!(g.mmio_read(R_FAULTED), 0);
}

/// Suspend with a live task, resume on a fresh device over the same RAM: the task is rebuilt
/// from its root word and runs again (6.24).
fn suspend_and_resume(setup: Setup) {
    let mut g = Guest::new(setup);
    g.bring_up();
    g.define_task(1);
    g.exec(1);
    let blob = g.in_slot(|| {
        g.dev().will_suspend();
        g.dev().finish_suspend()
    });
    let blob = blob.expect("a suspend blob");
    g.release();

    // A fresh device over the same RAM, as a migration's destination is.
    let (device, host) = with_slot(Some(g.slot.clone()), || {
        Guest::make_device(setup, &g.platform, &g.ram, &g.irqs)
    });
    if host.is_some() {
        g.host = host;
    }
    g.device = Some(device);
    let mut error = None;
    let ok = g.in_slot(|| g.dev().will_resume(Some(&blob), Some(&mut error)));
    assert!(ok, "willResume refused: {error:?}");
    g.in_slot(|| g.dev().did_resume(Hold::NONE));
    assert_eq!(g.mmio_read(R_VERSION), 245);
    assert_eq!(g.mmio_read(R_ENABLE), 1, "the root FIFO came back");
    // The ring carries on from the suspended counts.
    g.exec(1);
    assert_eq!(g.fills().len(), 2);
    assert_eq!(g.mmio_read(R_FAULTED), 0);
}

macro_rules! in_every_setup {
    ($($name:ident),* $(,)?) => {
        mod blocks {
            $(#[test] fn $name() { super::$name(super::Setup::Blocks) })*
        }
        mod map {
            $(#[test] fn $name() { super::$name(super::Setup::Map) })*
        }
        mod isolate {
            $(#[test] fn $name() { super::$name(super::Setup::Isolate) })*
        }
    };
}

in_every_setup!(
    bring_up_and_nop,
    task_and_exec,
    fault_and_resume,
    suspend_and_resume
);

// ---------------------------------------------------------------------------------------------
// Isolate only: the service's death at op K
// ---------------------------------------------------------------------------------------------

/// A death in the service's own code, scripted at a Metal call.
const SERVICE_DEATH: DeathSite = DeathSite {
    kind: reims_vgpu_core::raise::death::DeathKind::Abort,
    file: "service",
    line: 0,
};

/// The task's service dies at its K-th `fillBuffer:range:value:` (op K = 2): the first exec
/// completes, the second faults by the death (status 4), the dead task's next command faults, a
/// task beside it keeps working, and the task deleted and defined again gets a new service.
#[test]
fn isolate_service_death_at_op_k() {
    const K: u64 = 2;
    let mut g = Guest::new(Setup::Isolate);
    g.bring_up();
    g.define_task(1);
    g.define_task(2);
    let launched = g.platform.thread_launcher().launched();
    assert_eq!(launched.len(), 2, "one service per task");
    // Task 1's execs are the only fills so far.
    g.platform.script().after(
        "fillBuffer:range:value:",
        K - 1,
        Step::Die(Some(SERVICE_DEATH)),
    );

    // Op 1: completes.
    g.exec(1);
    assert!(launched[0].death().is_none());
    // Op K: the service dies; the exec faults.
    let (at, _) = g.push(OP_EXEC_INDIRECT3, &Guest::exec_payload(1));
    g.ring();
    assert_eq!(g.wait_fault(), at);
    eventually("the service's end", || launched[0].ended());
    assert_eq!(
        launched[0].death(),
        Some(SERVICE_DEATH),
        "the service died at op K"
    );
    g.resume();
    g.wait_stamp(g.stamp);

    // The dead task's next command that asks the service faults too.
    let (at, _) = g.push(OP_SET_OBJECT_LIST, &Guest::object_list_payload(1));
    g.ring();
    assert_eq!(g.wait_fault(), at);
    g.resume();
    g.wait_stamp(g.stamp);
    // MapMemory2 never contacts the service: it completes.
    g.run(OP_MAP_MEMORY2, &Guest::map_payload(1));
    assert_eq!(g.mmio_read(R_FAULTED), 0);

    // The task beside it keeps working.
    let before = g.fills().len();
    g.exec(2);
    assert_eq!(g.fills().len(), before + 1);
    assert!(launched[1].death().is_none());

    // Deleted and defined again, task 1 gets a new service and runs.
    let mut p = Vec::new();
    p.extend_from_slice(&1u32.to_le_bytes());
    g.run(OP_DELETE_TASK, &p);
    g.define_task(1);
    assert_eq!(g.platform.thread_launcher().launched().len(), 3);
    let before = g.fills().len();
    g.exec(1);
    assert_eq!(g.fills().len(), before + 1);
    assert_eq!(
        g.launched_deaths().iter().filter(|d| d.is_some()).count(),
        1,
        "only the first service died"
    );
}

/// A crash inside Metal under a task's service ends the process, as it does outside isolation.
#[test]
fn isolate_metal_crash_ends_the_process() {
    let mut g = Guest::new(Setup::Isolate);
    g.bring_up();
    g.define_task(1);
    g.platform
        .script()
        .push("fillBuffer:range:value:", Step::crash());
    let _ = g.push(OP_EXEC_INDIRECT3, &Guest::exec_payload(1));
    g.ring();
    eventually("the process's death", || g.slot.death().is_some());
    assert_eq!(
        g.slot.death(),
        Some(metal_crash_site("fillBuffer:range:value:"))
    );
    assert!(g.launched_deaths().iter().all(Option::is_none));
    // The process is dead: nothing of it is torn down.
    std::mem::forget(g);
}
