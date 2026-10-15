//! Fake: `FakeHost` (the host's callbacks, as the device reads them, as ordered recorders) and
//! `FakeHostQueue` (architecture 13.2).
//!
//! `FakeHost::blocks()` gives a full `HostBlocks`; a test removes a block (sets it to `None`) to
//! exercise core's nil-block paths. Each block logs one call (class `HostBlocks`, the block's name,
//! such as `mapMemory`, as selector) into the shared log, consults the script, and otherwise does
//! what QEMU does, over fake memory:
//! - `createTask` reserves the task's VA in the address space and returns a fresh task;
//! - `mapMemory` remaps the guest's physical ranges into the task's VA (read-only as asked);
//! - `unmapMemory` replaces the range with zero fill;
//! - `readMemory` copies guest-physical bytes into core's buffer;
//! - `addTraceRange` keeps the handler so a test can fire it ([`FakeHost::fire_trace`]).
//!
//! The host's own mach calls go to an unlogged view of the address space: they are QEMU's, not
//! core's platform calls.

use crate::num;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::sync::{Condvar, Mutex};

use super::calllog::{Arg, CallLog};
use super::guest_ram::{GuestRam, fill_host_buffer};
use super::platform::FakePlatform;
use super::script::{Script, Step};
use super::vm::FakeVm;
use crate::host::{
    HostBlocks, HostPtr, HostQueue, HostTask, HostTraceRange, PhysicalMemoryRange,
    TraceRangeHandler,
};
use crate::platform::ObjId;
use crate::platform::os::vm::{HostVm, VmProt};
use crate::rt::queue::{Executor, Job};

const CLASS: &str = "HostBlocks";

/// One task `createTask` made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FakeTask {
    /// The opaque task value core holds.
    pub task: HostTask,
    /// Its VA base (`baseAddress`).
    pub base: HostPtr,
    /// Its VA size.
    pub size: u64,
}

/// The host: QEMU's blocks over fake guest RAM.
pub struct FakeHost {
    log: CallLog,
    script: Script,
    vm: FakeVm,
    ram: GuestRam,
    next: AtomicUsize,
    tasks: Mutex<Vec<FakeTask>>,
    traces: Mutex<Vec<(HostTraceRange, PhysicalMemoryRange, TraceRangeHandler)>>,
    interrupts: Mutex<Vec<u32>>,
}

impl std::fmt::Debug for FakeHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeHost")
            .field("tasks", &self.tasks.lock().len())
            .finish_non_exhaustive()
    }
}

impl FakeHost {
    /// A host over `ram`, logging into `platform`'s log and scripted by its script. The RAM is
    /// adopted into the platform's address space, as QEMU's RAM is mapped in its process.
    pub fn new(platform: &FakePlatform, ram: GuestRam) -> Arc<FakeHost> {
        platform.fake_vm().adopt(&ram);
        Arc::new(FakeHost {
            log: platform.call_log().clone(),
            script: platform.script().clone(),
            vm: platform.fake_vm().unlogged(),
            ram,
            next: AtomicUsize::new(0x1000),
            tasks: Mutex::new(Vec::new()),
            traces: Mutex::new(Vec::new()),
            interrupts: Mutex::new(Vec::new()),
        })
    }

    /// The guest RAM.
    pub fn ram(&self) -> &GuestRam {
        &self.ram
    }

    fn call(&self, sel: &'static str, args: Vec<Arg>) -> Step {
        self.log.push(ObjId(0), CLASS, sel, args);
        let step = self.script.next(ObjId(0), sel);
        step.getter(sel);
        step
    }

    /// [`FakeHost::call`] for the five memory blocks: a scripted `Step::Raise` is the host's
    /// exception, which ends the process at the call (`reims_vgpu_c`'s `ffi::host_call`).
    fn call_block(&self, sel: &'static str, args: Vec<Arg>) -> Step {
        self.log.push(ObjId(0), CLASS, sel, args);
        let step = self.script.next(ObjId(0), sel);
        if matches!(step, Step::Raise { .. }) {
            crate::raise::death::fatal(crate::raise::death::DeathId::HostRaised { call: sel })
        }
        step
    }

    fn next_opaque(&self) -> usize {
        self.next.fetch_add(0x10, Ordering::SeqCst)
    }

    /// Every vector `raiseInterrupt` received, in order.
    pub fn interrupts(&self) -> Vec<u32> {
        self.interrupts.lock().clone()
    }

    /// The live tasks.
    pub fn tasks(&self) -> Vec<FakeTask> {
        self.tasks.lock().clone()
    }

    /// The live trace ranges.
    pub fn trace_ranges(&self) -> Vec<(HostTraceRange, PhysicalMemoryRange)> {
        self.traces
            .lock()
            .iter()
            .map(|(h, r, _)| (*h, *r))
            .collect()
    }

    /// Calls the handler of trace range `h` with `dirty` (QEMU noticing a guest write). `false`
    /// if no such range is live.
    pub fn fire_trace(&self, h: HostTraceRange, dirty: PhysicalMemoryRange) -> bool {
        let handler = self
            .traces
            .lock()
            .iter()
            .find(|(x, _, _)| *x == h)
            .map(|(_, _, f)| f.clone());
        match handler {
            Some(f) => {
                f(dirty);
                true
            }
            None => false,
        }
    }

    fn create_task(&self, vm_size: u64, base: &mut HostPtr) -> HostTask {
        let step = self.call_block("createTask", vec![Arg::U64(vm_size)]);
        if step.is_nil() {
            return HostTask::NULL;
        }
        let Ok(addr) = self.vm.allocate_anywhere(vm_size) else {
            return HostTask::NULL;
        };
        *base = addr;
        let task = HostTask::from_addr(self.next_opaque());
        self.tasks.lock().push(FakeTask {
            task,
            base: addr,
            size: vm_size,
        });
        task
    }

    fn destroy_task(&self, task: HostTask) {
        let _ = self.call_block("destroyTask", vec![Arg::Ptr(task.addr())]);
        let mut g = self.tasks.lock();
        if let Some(p) = g.iter().position(|t| t.task == task) {
            let t = g.remove(p);
            drop(g);
            let _ = self.vm.deallocate(t.base, t.size);
        }
    }

    fn task(&self, task: HostTask) -> Option<FakeTask> {
        self.tasks.lock().iter().find(|t| t.task == task).copied()
    }

    fn map_memory(
        &self,
        task: HostTask,
        count: u32,
        offset: u64,
        read_only: bool,
        ranges: &[PhysicalMemoryRange],
    ) -> bool {
        let step = self.call_block(
            "mapMemory",
            vec![
                Arg::Ptr(task.addr()),
                Arg::U32(count),
                Arg::U64(offset),
                Arg::Bool(read_only),
                Arg::val(ranges),
            ],
        );
        if let Some(b) = step.get::<bool>() {
            return b;
        }
        let Some(t) = self.task(task) else {
            return false;
        };
        let mut at = t.base.wrapping_add(offset);
        for r in ranges {
            let Some(src) = self.ram.host_addr(r.physical_address) else {
                return false;
            };
            if self
                .vm
                .remap_shared_overwrite(at, src, r.physical_length)
                .is_err()
            {
                return false;
            }
            if read_only
                && self
                    .vm
                    .protect_max(at, r.physical_length, VmProt::READ)
                    .is_err()
            {
                return false;
            }
            at = at.wrapping_add(r.physical_length);
        }
        true
    }

    fn unmap_memory(&self, task: HostTask, offset: u64, length: u64) -> bool {
        let step = self.call_block(
            "unmapMemory",
            vec![Arg::Ptr(task.addr()), Arg::U64(offset), Arg::U64(length)],
        );
        if let Some(b) = step.get::<bool>() {
            return b;
        }
        let Some(t) = self.task(task) else {
            return false;
        };
        self.vm
            .allocate_fixed_overwrite(t.base.wrapping_add(offset), length)
            .is_ok()
    }

    fn read_memory(&self, pa: u64, length: u64, dst: HostPtr) -> bool {
        let step = self.call_block(
            "readMemory",
            vec![Arg::U64(pa), Arg::U64(length), Arg::Ptr(dst.addr())],
        );
        if let Some(b) = step.get::<bool>() {
            return b;
        }
        self.vm.sync();
        match self.ram.read_vec(pa, length) {
            Some(bytes) => {
                fill_host_buffer(dst, &bytes);
                true
            }
            None => false,
        }
    }

    fn add_trace_range(&self, range: PhysicalMemoryRange, h: TraceRangeHandler) -> HostTraceRange {
        let step = self.call("addTraceRange", vec![Arg::val(&range)]);
        if step.is_nil() {
            return HostTraceRange::NULL;
        }
        let r = HostTraceRange::from_addr(self.next_opaque());
        self.traces.lock().push((r, range, h));
        r
    }

    fn remove_trace_range(&self, r: HostTraceRange) {
        let _ = self.call("removeTraceRange", vec![Arg::Ptr(r.addr())]);
        self.traces.lock().retain(|(x, _, _)| *x != r);
    }

    fn raise_interrupt(&self, v: u32) {
        let _ = self.call("raiseInterrupt", vec![Arg::U32(v)]);
        self.interrupts.lock().push(v);
    }

    /// All eight blocks.
    pub fn blocks(self: &Arc<Self>) -> HostBlocks {
        let (a, b, c, d, e, f, g, h) = (
            self.clone(),
            self.clone(),
            self.clone(),
            self.clone(),
            self.clone(),
            self.clone(),
            self.clone(),
            self.clone(),
        );
        HostBlocks {
            raise_interrupt: Some(Arc::new(move |v| a.raise_interrupt(v))),
            create_task: Some(Arc::new(move |size, base| b.create_task(size, base))),
            destroy_task: Some(Arc::new(move |t| c.destroy_task(t))),
            map_memory: Some(Arc::new(move |t, n, off, ro, r| {
                d.map_memory(t, n, off, ro, r)
            })),
            unmap_memory: Some(Arc::new(move |t, off, len| e.unmap_memory(t, off, len))),
            read_memory: Some(Arc::new(move |pa, len, dst| f.read_memory(pa, len, dst))),
            add_trace_range: Some(Arc::new(move |r, hd| g.add_trace_range(r, hd))),
            remove_trace_range: Some(Arc::new(move |r| h.remove_trace_range(r))),
        }
    }
}

/// How a [`FakeHostQueue`] runs what it is given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueMode {
    /// At once, on the submitting thread.
    Inline,
    /// Held until the test calls [`FakeHostQueue::run_pending`] (on the test's thread).
    Manual,
    /// In order on one test-owned worker thread.
    Thread,
}

#[derive(Default)]
struct QueueState {
    jobs: VecDeque<Job>,
    closed: bool,
}

/// A display's queue, as a test's host gives it: handlers run on it in submission order (7.1),
/// and each submission is counted.
pub struct FakeHostQueue {
    mode: QueueMode,
    state: Arc<(Mutex<QueueState>, Condvar)>,
    submitted: AtomicUsize,
    /// `self`, for [`HostQueue::dispatch_retain`] (a counted clone).
    me: std::sync::Weak<FakeHostQueue>,
    /// `dispatch_retain`s so far.
    dispatch_retains: AtomicUsize,
}

impl std::fmt::Debug for FakeHostQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeHostQueue")
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

impl FakeHostQueue {
    /// A queue running jobs as `mode` says.
    pub fn new(mode: QueueMode) -> Arc<FakeHostQueue> {
        let q = Arc::new_cyclic(|me| FakeHostQueue {
            mode,
            state: Arc::new((Mutex::new(QueueState::default()), Condvar::new())),
            submitted: AtomicUsize::new(0),
            me: me.clone(),
            dispatch_retains: AtomicUsize::new(0),
        });
        if mode == QueueMode::Thread {
            let st = q.state.clone();
            let slot = crate::testkit::capture::current_slot();
            let r = std::thread::Builder::new()
                .name("fake-host-queue".into())
                .spawn(move || {
                    crate::testkit::capture::with_slot(slot, || {
                        loop {
                            let job = {
                                let mut g = st.0.lock();
                                loop {
                                    if let Some(j) = g.jobs.pop_front() {
                                        break Some(j);
                                    }
                                    if g.closed {
                                        break None;
                                    }
                                    st.1.wait(&mut g);
                                }
                            };
                            match job {
                                Some(j) => j(),
                                None => return,
                            }
                        }
                    })
                });
            if r.is_err() {
                crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "fake/host.rs")
            }
        }
        q
    }

    /// Runs every held job in order (manual mode); returns how many ran.
    pub fn run_pending(&self) -> usize {
        let mut n = 0usize;
        loop {
            let job = self.state.0.lock().jobs.pop_front();
            match job {
                Some(j) => {
                    j();
                    n = n.wrapping_add(1);
                }
                None => return n,
            }
        }
    }

    /// `dispatch_retain`s so far ([`HostQueue::dispatch_retain`]).
    pub fn dispatch_retains(&self) -> usize {
        self.dispatch_retains.load(Ordering::SeqCst)
    }

    /// Jobs submitted so far.
    pub fn submitted(&self) -> usize {
        self.submitted.load(Ordering::SeqCst)
    }

    /// Jobs not yet run.
    pub fn pending(&self) -> usize {
        self.state.0.lock().jobs.len()
    }

    /// Blocks (real time, up to 10 s) until the worker has run every submitted job.
    pub fn drain(&self) -> bool {
        for _ in 0..10_000 {
            if self.pending() == 0 {
                // One more slice lets the job the worker popped finish.
                std::thread::sleep(std::time::Duration::from_millis(2));
                if self.pending() == 0 {
                    return true;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        false
    }
}

impl Drop for FakeHostQueue {
    fn drop(&mut self) {
        self.state.0.lock().closed = true;
        self.state.1.notify_all();
    }
}

impl Executor for FakeHostQueue {
    fn exec_async(&self, job: Job) {
        self.submitted.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            QueueMode::Inline => job(),
            QueueMode::Manual | QueueMode::Thread => {
                self.state.0.lock().jobs.push_back(job);
                self.state.1.notify_all();
            }
        }
    }
}

/// Core's `CoalescingSignal` models the display's newFrame source on it, as libdispatch runs one on
/// a queue not suspended at the source's creation (the first merge gets its own slot, at the
/// merge). The fake has no suspension.
impl HostQueue for FakeHostQueue {
    /// A counted clone: the same queue, one more reference.
    fn dispatch_retain(&self) -> Arc<dyn HostQueue> {
        self.dispatch_retains.fetch_add(1, Ordering::SeqCst);
        match self.me.upgrade() {
            Some(q) => q,
            // Only reachable from the queue's own drop.
            None => {
                crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "fake/host.rs")
            }
        }
    }
}

/// The total length of a range list (for tests building `mapMemory` calls).
pub fn total_length(ranges: &[PhysicalMemoryRange]) -> u64 {
    ranges
        .iter()
        .fold(0u64, |a, r| a.wrapping_add(r.physical_length))
}

/// A range list's length in pages of `page`.
pub fn pages(ranges: &[PhysicalMemoryRange], page: u64) -> u64 {
    num::div0(total_length(ranges), page.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host() -> (FakePlatform, Arc<FakeHost>) {
        let p = FakePlatform::new();
        let ram = GuestRam::with_range(0x1_0000_0000, 0x10000).unwrap();
        let h = FakeHost::new(&p, ram);
        (p, h)
    }

    /// Interrupts are recorded in order and logged
    #[test]
    fn raise_interrupt() {
        let (p, h) = host();
        let b = h.blocks();
        let f = b.raise_interrupt.unwrap();
        f(1);
        f(3);
        assert_eq!(h.interrupts(), vec![1, 3]);
        assert_eq!(
            p.call_log().calls_to("raiseInterrupt")[1].args,
            vec![Arg::U32(3)]
        );
    }

    /// createTask, mapMemory, readMemory through the task's VA, unmap, destroy
    #[test]
    fn task_lifecycle() {
        let (p, h) = host();
        let b = h.blocks();
        let mut base = HostPtr::NULL;
        let t = (b.create_task.clone().unwrap())(0x10000, &mut base);
        assert!(!t.is_null() && !base.is_null());
        h.ram().write_u32(0x1_0000_4000, 0xabcd);
        let ranges = [PhysicalMemoryRange {
            physical_address: 0x1_0000_4000,
            physical_length: 0x4000,
        }];
        assert!((b.map_memory.clone().unwrap())(t, 1, 0x8000, true, &ranges));
        let vm = p.fake_vm();
        assert_eq!(
            vm.read(base.wrapping_add(0x8000), 4).unwrap(),
            0xabcdu32.to_le_bytes()
        );
        assert!(!vm.is_writable(base.wrapping_add(0x8000), 1));
        let mut buf = [0u8; 4];
        assert!((b.read_memory.clone().unwrap())(
            0x1_0000_4000,
            4,
            HostPtr::from_ptr(buf.as_mut_ptr())
        ));
        assert_eq!(u32::from_le_bytes(buf), 0xabcd);
        assert!(!(b.read_memory.clone().unwrap())(
            0x9_0000_0000,
            4,
            HostPtr::from_ptr(buf.as_mut_ptr())
        ));
        assert!((b.unmap_memory.clone().unwrap())(t, 0x8000, 0x4000));
        assert_eq!(vm.read(base.wrapping_add(0x8000), 4).unwrap(), vec![0; 4]);
        (b.destroy_task.clone().unwrap())(t);
        assert!(h.tasks().is_empty());
        assert!(!(b.unmap_memory.unwrap())(t, 0, 1));
        // only the blocks were logged, not the host's own mach calls
        assert_eq!(p.call_log().count("mach_vm_remap"), 0);
    }

    /// Scripted block results
    #[test]
    fn scripted() {
        let (p, h) = host();
        let b = h.blocks();
        p.script().push("createTask", Step::Nil);
        let mut base = HostPtr::NULL;
        assert!((b.create_task.unwrap())(0x4000, &mut base).is_null());
        p.script().push("mapMemory", Step::value(false));
        assert!(!(b.map_memory.clone().unwrap())(
            HostTask::NULL,
            0,
            0,
            false,
            &[]
        ));
        // A raise is fatal at the call.
        {
            p.script()
                .push("mapMemory", Step::raise("HostMapFailed", "refused"));
            let d = crate::testkit::capture::catch_death(|| {
                let _ = (b.map_memory.clone().unwrap())(HostTask::NULL, 0, 0, false, &[]);
            })
            .unwrap_err();
            assert_eq!(
                (d.kind, d.file),
                (crate::raise::death::DeathKind::HostRaised, "mapMemory")
            );
        }
    }

    /// Trace ranges keep their handler until removed
    #[test]
    fn trace_ranges() {
        let (_, h) = host();
        let b = h.blocks();
        let hits = Arc::new(Mutex::new(Vec::new()));
        let h2 = hits.clone();
        let r = PhysicalMemoryRange {
            physical_address: 0x1000,
            physical_length: 0x100,
        };
        let id = (b.add_trace_range.unwrap())(r, Arc::new(move |d| h2.lock().push(d)));
        assert_eq!(h.trace_ranges(), vec![(id, r)]);
        assert!(h.fire_trace(id, r));
        assert_eq!(*hits.lock(), vec![r]);
        (b.remove_trace_range.unwrap())(id);
        assert!(!h.fire_trace(id, r));
        assert_eq!(total_length(&[r, r]), 0x200);
        assert_eq!(pages(&[r], 0x100), 1);
    }

    /// The three queue modes keep order
    #[test]
    fn queues() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        for mode in [QueueMode::Inline, QueueMode::Manual, QueueMode::Thread] {
            seen.lock().clear();
            let q = FakeHostQueue::new(mode);
            for i in 0..3 {
                let s = seen.clone();
                q.exec_async(Box::new(move || s.lock().push(i)));
            }
            match mode {
                QueueMode::Inline => assert_eq!(q.pending(), 0),
                QueueMode::Manual => {
                    assert_eq!(q.pending(), 3);
                    assert_eq!(q.run_pending(), 3);
                }
                QueueMode::Thread => assert!(q.drain()),
            }
            assert_eq!(*seen.lock(), vec![0, 1, 2]);
            assert_eq!(q.submitted(), 3);
        }
    }
}
