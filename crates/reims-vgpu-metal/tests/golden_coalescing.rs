//! Golden test: `CoalescingSignal` against dispatch DATA_ADD/DATA_OR sources.
//!
//! Each scenario runs twice on the same kind of target, a serial dispatch queue that starts
//! suspended (or, for the `busy` ones, is kept busy by a block of its own): once with a real
//! `dispatch_source` and once with `CoalescingSignal` whose executor is that queue. Merges made
//! while the queue cannot run must coalesce identically; the handler's data sequence (and, for
//! the order scenarios, its slot among the queue's other blocks) must match. How many calls a
//! burst gives under free-running timing is timing, so it is never compared: every burst here is
//! made while the target cannot run. The one thing `CoalescingSignal` does not model is a target
//! suspended at the source's creation (`first_fire_order_before_activation`, dispatch only).

#![cfg(target_os = "macos")] // Metal is macOS's: the crate is empty elsewhere
#![allow(clippy::unwrap_used, clippy::panic)] // A failure is the assertion

use std::sync::Arc;
use std::time::Duration;

use block2::RcBlock;
use dispatch2::{
    _dispatch_source_type_data_add, _dispatch_source_type_data_or, DispatchObject, DispatchQueue,
    DispatchRetained, DispatchSource,
};
use reims_vgpu_core::rt::{CoalescingSignal, Executor, Job};
use reims_vgpu_core::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Add,
    Or,
}

#[derive(Debug, Clone, Copy)]
enum Step {
    /// `dispatch_source_merge_data`.
    Merge(u64),
    /// `dispatch_source_cancel`.
    Cancel,
    /// `dispatch_resume` of the target queue (it starts suspended).
    Resume,
    /// `dispatch_suspend` of the target queue.
    Suspend,
    /// Let the target run everything queued.
    Drain,
}

/// The handler merges these into its own source on its first call (a merge while the handler
/// runs).
type Reentrant = &'static [u64];

/// `dispatch_async` onto a real dispatch queue.
struct DispatchExec(DispatchRetained<DispatchQueue>);
impl Executor for DispatchExec {
    fn exec_async(&self, job: Job) {
        self.0.exec_async(job);
    }
}

fn drain(q: &DispatchQueue) {
    // A fire may resubmit from inside the handler; let the queue settle.
    for _ in 0..4 {
        q.exec_sync(|| ());
        std::thread::sleep(Duration::from_millis(5));
    }
    q.exec_sync(|| ());
}

fn run_script(q: &DispatchQueue, script: &[Step], merge: &dyn Fn(u64), cancel: &dyn Fn()) {
    let mut suspended = true;
    for step in script {
        match *step {
            Step::Merge(v) => merge(v),
            Step::Cancel => cancel(),
            Step::Resume => {
                q.resume();
                suspended = false;
            }
            Step::Suspend => {
                q.suspend();
                suspended = true;
            }
            Step::Drain => drain(q),
        }
    }
    if suspended {
        q.resume();
    }
    drain(q);
}

fn real(kind: Kind, script: &[Step], reentrant: Reentrant) -> Vec<u64> {
    let q = DispatchQueue::new("golden.coalescing.real", None);
    q.suspend();
    let ty = match kind {
        // SAFETY: the address of libdispatch's source type descriptor, as DISPATCH_SOURCE_TYPE_*.
        Kind::Add => (&raw const _dispatch_source_type_data_add).cast_mut(),
        Kind::Or => (&raw const _dispatch_source_type_data_or).cast_mut(),
    };
    // SAFETY: a DATA_ADD/DATA_OR type takes handle 0 and mask 0; `q` is a live queue.
    let source = unsafe { DispatchSource::new(ty, 0, 0, Some(&q)) };
    let log = Arc::new(Mutex::new(Vec::new()));
    let src_ptr: *const DispatchSource = &*source;
    let l = log.clone();
    let block = RcBlock::new(move || {
        // SAFETY: the source outlives every handler call: it is cancelled and its queue drained
        // before `source` drops.
        let s = unsafe { &*src_ptr };
        let d = u64::try_from(s.data()).unwrap();
        let first = {
            let mut l = l.lock();
            l.push(d);
            l.len() == 1
        };
        if first {
            for &v in reentrant {
                s.merge_data(usize::try_from(v).unwrap());
            }
        }
    });
    // SAFETY: the block is a valid `void (^)(void)`; libdispatch copies it.
    unsafe { source.set_event_handler_with_block(RcBlock::as_ptr(&block)) };
    source.resume();
    run_script(
        &q,
        script,
        &|v| source.merge_data(usize::try_from(v).unwrap()),
        &|| source.cancel(),
    );
    source.cancel();
    drain(&q);
    log.lock().clone()
}

fn coalescing(kind: Kind, script: &[Step], reentrant: Reentrant) -> Vec<u64> {
    let q = DispatchQueue::new("golden.coalescing.signal", None);
    q.suspend();
    let exec: Arc<dyn Executor> = Arc::new(DispatchExec(q.clone()));
    let log = Arc::new(Mutex::new(Vec::new()));
    let slot: Arc<Mutex<Option<CoalescingSignal>>> = Arc::new(Mutex::new(None));
    let (l, s2) = (log.clone(), slot.clone());
    let handler = move |d: u64| {
        let first = {
            let mut l = l.lock();
            l.push(d);
            l.len() == 1
        };
        if first && let Some(sig) = s2.lock().clone() {
            for &v in reentrant {
                sig.merge(v);
            }
        }
    };
    let sig = match kind {
        Kind::Add => CoalescingSignal::new_add(exec, handler),
        Kind::Or => CoalescingSignal::new_or(exec, handler),
    };
    *slot.lock() = Some(sig.clone());
    run_script(&q, script, &|v| sig.merge(v), &|| sig.cancel());
    sig.cancel();
    drain(&q);
    *slot.lock() = None;
    log.lock().clone()
}

fn same(kind: Kind, script: &[Step], reentrant: Reentrant, expect: &[u64]) {
    let r = real(kind, script, reentrant);
    let p = coalescing(kind, script, reentrant);
    assert_eq!(r, expect, "dispatch_source {kind:?} {script:?}");
    assert_eq!(p, r, "CoalescingSignal {kind:?} {script:?}");
}

use Step::*;

/// ADD merges made before the target runs give one call with the sum
#[test]
fn add_coalesces() {
    same(Kind::Add, &[Merge(1), Merge(1), Merge(3)], &[], &[5]);
}

/// OR merges made before the target runs give one call with the bits
#[test]
fn or_coalesces() {
    same(Kind::Or, &[Merge(1), Merge(1), Merge(4)], &[], &[5]);
}

/// A merge of 0 never submits the handler (ADD and OR)
#[test]
fn zero_merge_is_noop() {
    same(Kind::Add, &[Merge(0)], &[], &[]);
    same(Kind::Or, &[Merge(0)], &[], &[]);
}

/// An ADD burst whose sum wraps to 0 calls nothing; a later merge calls with its value
#[test]
fn add_wrapping_to_zero() {
    same(
        Kind::Add,
        &[Merge(u64::MAX), Merge(1), Resume, Drain, Suspend, Merge(2)],
        &[],
        &[2],
    );
}

/// A merge after the handler ran schedules a new call
#[test]
fn merge_after_fire_schedules_again() {
    same(
        Kind::Add,
        &[Merge(1), Resume, Drain, Suspend, Merge(2), Merge(5)],
        &[],
        &[1, 7],
    );
}

/// Merges while the handler runs give exactly one more call, after it
#[test]
fn merge_during_handler_runs_after() {
    same(Kind::Add, &[Merge(1)], &[10, 20], &[1, 30]);
    same(Kind::Or, &[Merge(1)], &[2, 4], &[1, 6]);
}

/// Cancel before the target runs drops the pending data; later merges do nothing
#[test]
fn cancel_stops_handler() {
    same(
        Kind::Add,
        &[Merge(1), Cancel, Resume, Drain, Merge(1)],
        &[],
        &[],
    );
    same(Kind::Or, &[Merge(1), Cancel, Merge(2)], &[], &[]);
}

/// Where the first handler call lands among other jobs on the target. The source is created and
/// resumed on a suspended target; with `settle`, the target then runs once (the source's
/// activation) and is suspended again. Then `P` is queued, 1 is merged, `after` is queued, and
/// the target runs. The log holds the handler calls (`h<data>`) and the foreign jobs in the
/// order the target ran them.
fn order_real(settle: bool) -> Vec<String> {
    let q = DispatchQueue::new("golden.coalescing.order.real", None);
    q.suspend();
    // SAFETY: the address of libdispatch's DATA_ADD source type descriptor.
    let ty = (&raw const _dispatch_source_type_data_add).cast_mut();
    // SAFETY: a DATA_ADD type takes handle 0 and mask 0; `q` is a live queue.
    let source = unsafe { DispatchSource::new(ty, 0, 0, Some(&q)) };
    let log = Arc::new(Mutex::new(Vec::<String>::new()));
    let src_ptr: *const DispatchSource = &*source;
    let l = log.clone();
    let block = RcBlock::new(move || {
        // SAFETY: the source is cancelled and its queue drained before `source` drops.
        let d = unsafe { &*src_ptr }.data();
        l.lock().push(format!("h{d}"));
    });
    // SAFETY: the block is a valid `void (^)(void)`; libdispatch copies it.
    unsafe { source.set_event_handler_with_block(RcBlock::as_ptr(&block)) };
    source.resume();
    if settle {
        q.resume();
        drain(&q);
        q.suspend();
    }
    let l = log.clone();
    q.exec_async(move || l.lock().push("P".into()));
    source.merge_data(1);
    let l = log.clone();
    q.exec_async(move || l.lock().push("after".into()));
    q.resume();
    drain(&q);
    source.cancel();
    drain(&q);
    log.lock().clone()
}

fn order_coalescing(settle: bool) -> Vec<String> {
    let q = DispatchQueue::new("golden.coalescing.order.signal", None);
    q.suspend();
    let exec: Arc<dyn Executor> = Arc::new(DispatchExec(q.clone()));
    let log = Arc::new(Mutex::new(Vec::<String>::new()));
    let l = log.clone();
    let sig = CoalescingSignal::new_add(exec, move |d| l.lock().push(format!("h{d}")));
    if settle {
        q.resume();
        drain(&q);
        q.suspend();
    }
    let l = log.clone();
    q.exec_async(move || l.lock().push("P".into()));
    sig.merge(1);
    let l = log.clone();
    q.exec_async(move || l.lock().push("after".into()));
    q.resume();
    drain(&q);
    sig.cancel();
    drain(&q);
    log.lock().clone()
}

/// Once the source's activation has run on the target, the first handler call is queued at merge
/// time, behind jobs queued before the merge (dispatch and `CoalescingSignal` agree)
#[test]
fn first_fire_order_after_activation() {
    let r = order_real(true);
    assert_eq!(r, ["P", "h1", "after"], "dispatch_source");
    assert_eq!(order_coalescing(true), r, "CoalescingSignal");
}

/// On a target suspended since `dispatch_resume` of the source, dispatch runs the first handler
/// call in the activation's slot, ahead of jobs queued between the resume and the merge.
/// `CoalescingSignal` does not model it (rt/signal.rs): core never suspends the queues a display's
/// newFrame source targets.
#[test]
fn first_fire_order_before_activation() {
    assert_eq!(order_real(false), ["h1", "P", "after"], "dispatch_source");
}

/// A serial queue kept busy by a block that waits for the returned sender.
fn busy_queue(label: &str) -> (DispatchRetained<DispatchQueue>, std::sync::mpsc::Sender<()>) {
    let q = DispatchQueue::new(label, None);
    let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
    let (in_tx, in_rx) = std::sync::mpsc::channel::<()>();
    q.exec_async(move || {
        in_tx.send(()).unwrap();
        go_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    });
    in_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    (q, go_tx)
}

/// Logs `~h` when dropped: what a source's handler block captures, freed with it.
struct Freed(Arc<Mutex<Vec<String>>>);
impl Drop for Freed {
    fn drop(&mut self) {
        self.0.lock().push("~h".into());
    }
}

/// A real DATA_ADD source on `q`, resumed, whose handler logs `h<data>` and whose freed handler
/// logs `~h`.
fn real_source(
    q: &DispatchQueue,
    log: &Arc<Mutex<Vec<String>>>,
) -> DispatchRetained<DispatchSource> {
    // SAFETY: the address of libdispatch's DATA_ADD source type descriptor.
    let ty = (&raw const _dispatch_source_type_data_add).cast_mut();
    // SAFETY: a DATA_ADD type takes handle 0 and mask 0; `q` is a live queue.
    let source = unsafe { DispatchSource::new(ty, 0, 0, Some(q)) };
    let src_ptr: *const DispatchSource = &*source;
    let (l, freed) = (log.clone(), Freed(log.clone()));
    let block = RcBlock::new(move || {
        let _ = &freed;
        // SAFETY: the source outlives its handler calls (it is released after the drain).
        let d = unsafe { &*src_ptr }.data();
        l.lock().push(format!("h{d}"));
    });
    // SAFETY: the block is a valid `void (^)(void)`; libdispatch copies it.
    unsafe { source.set_event_handler_with_block(RcBlock::as_ptr(&block)) };
    drop(block);
    source.resume();
    source
}

/// A `CoalescingSignal` on `q` standing for [`real_source`]'s source, with the same handler.
fn coalescing_source(
    q: &DispatchRetained<DispatchQueue>,
    log: &Arc<Mutex<Vec<String>>>,
) -> CoalescingSignal {
    let exec: Arc<dyn Executor> = Arc::new(DispatchExec(q.clone()));
    let (l, freed) = (log.clone(), Freed(log.clone()));
    CoalescingSignal::new_add(exec, move |d| {
        let _ = &freed;
        l.lock().push(format!("h{d}"));
    })
}

fn mark(q: &DispatchQueue, log: &Arc<Mutex<Vec<String>>>, name: &'static str) {
    let l = log.clone();
    q.exec_async(move || l.lock().push(name.into()));
}

/// A source created and resumed while its target is busy (not suspended) takes no slot: the first
/// handler call is queued at the merge, behind the jobs queued before it (dispatch and
/// `CoalescingSignal` agree: `P h1 after`)
#[test]
fn first_fire_order_on_a_busy_target() {
    let run = |real: bool| {
        let (q, go) = busy_queue("golden.coalescing.busy");
        let log = Arc::new(Mutex::new(Vec::<String>::new()));
        let (r, p) = if real {
            (Some(real_source(&q, &log)), None)
        } else {
            (None, Some(coalescing_source(&q, &log)))
        };
        mark(&q, &log, "P");
        if let Some(s) = &r {
            s.merge_data(1);
        }
        if let Some(s) = &p {
            s.merge(1);
        }
        mark(&q, &log, "after");
        go.send(()).unwrap();
        drain(&q);
        if let Some(s) = &r {
            s.cancel();
        }
        if let Some(s) = &p {
            s.cancel();
        }
        drain(&q);
        drop((r, p));
        drain(&q);
        let got = log.lock().clone();
        got.into_iter().filter(|e| e != "~h").collect::<Vec<_>>()
    };
    let r = run(true);
    assert_eq!(r, ["P", "h1", "after"], "dispatch_source");
    assert_eq!(run(false), r, "CoalescingSignal");
}

/// A source cancelled on a busy target before any merge frees its handler in a callout queued at
/// the cancel, behind the jobs queued before it (dispatch and `CoalescingSignal` agree: `A ~h B`)
#[test]
fn cancel_on_a_busy_target() {
    let run = |real: bool| {
        let (q, go) = busy_queue("golden.coalescing.busy.cancel");
        let log = Arc::new(Mutex::new(Vec::<String>::new()));
        let (r, p) = if real {
            (Some(real_source(&q, &log)), None)
        } else {
            (None, Some(coalescing_source(&q, &log)))
        };
        mark(&q, &log, "A");
        if let Some(s) = &r {
            s.cancel();
        }
        if let Some(s) = &p {
            s.cancel();
        }
        mark(&q, &log, "B");
        go.send(()).unwrap();
        drain(&q);
        drop((r, p));
        drain(&q);
        log.lock().clone()
    };
    let r = run(true);
    assert_eq!(r, ["A", "~h", "B"], "dispatch_source");
    assert_eq!(run(false), r, "CoalescingSignal");
}
