"""GetComputeInfo (opcode 59): a compute pipeline's limits, written by the
host into the guest's reply buffer (docs/testing.md, sections 6.10, W10).

Payload {u32 taskID, u32 pipeline ref, u32 maxKey, u32 count, u64 reply task
VA}. The reply is count pairs of {u32 key, u32 value}:
{1, maxTotalThreadsPerThreadgroup} if maxKey > 1, {3, threadExecutionWidth}
if maxKey > 3, {4, staticThreadgroupMemoryLength} if maxKey > 4, then {0, 0}
if room is left. The values report the host's GPU, so these scenarios are
keyed by the host. Reply buffers are filled with 0xee first, so the dumps
show exactly what was written."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.compute_objects import Rig
from pgtest.registry import scenario
from pgtest.task import GPUTask


def info(s, tid, pso, max_key, count, offset, **kw):
    return s.submit(abi.OP_GET_COMPUTE_INFO, struct.pack("<IIIIQ", tid, pso, max_key, count, offset), **kw)


@scenario(host=True)
def compute_info_keys():
    """Every combination of maxKey and count that decides a branch, for
    double_gid: maxKey 5, 4, 3, 1 and 0 with room for 4 pairs, and maxKey 5
    with room for 3, 2 and 1. Then the pipeline's own values: a descriptor
    with maxTotalThreadsPerThreadgroup 64 (blob key 8), and tg_static's 256
    bytes of static threadgroup memory."""
    r = Rig("compute_info_keys", compute_info_keys.__doc__, host=True)
    reply = r.t.reserve(PAGE)
    pso = r.kernel()
    pso64 = r.kernel(max_threads=64)
    pso_tg = r.kernel("shaders/compute_tgstatic.metallib")
    r.t.fill(reply, 0x400, 0xee)
    r.ready()
    cases = [(pso, 5, 4), (pso, 4, 4), (pso, 3, 4), (pso, 1, 4), (pso, 0, 4),
             (pso, 5, 3), (pso, 5, 2), (pso, 5, 1), (pso64, 5, 4), (pso_tg, 5, 4)]
    for i, (p, mk, n) in enumerate(cases):
        info(r.s, r.t.tid, p, mk, n, reply + 0x40 * i, label="pso %d maxKey %d count %d" % (p, mk, n))
    r.dump(reply, 0x40 * len(cases), "replies, 0x40 apart")
    return r.end()


def either(s, irqs, payload, label):
    """A command that faults in some setups and completes in others: wait
    for its interrupt (a stamp and a fault raise one each), record the fault
    state, resume, and wait for the stamp, which a resume publishes."""
    st = s.fifo.cmd(abi.OP_GET_COMPUTE_INFO, payload)
    s.doorbell()
    s.wait(irqs=irqs, label=label + ": its interrupt")
    s.fault_state(label)
    s.resume_root()
    s.fifo.wait_stamp(st, label=label + ": its stamp, after the resume")


def count_zero(name, doc, off):
    r = Rig(name, doc, host=True)
    reply = r.t.reserve(PAGE)
    pso = r.kernel()
    r.t.fill(reply, 0x20, 0xee)
    r.ready()
    either(r.s, r.irqs + 1, struct.pack("<IIIIQ", r.t.tid, pso, 5, 0, reply + off), "count 0")
    r.dump(reply, 0x20, "reply")
    return r.end()


@scenario(host=True)
def compute_info_count_zero():
    """count 0, the reply offset 0x10 into a page: the memory-map setups map
    that one page, write nothing and complete; the blocks setup faults."""
    return count_zero("compute_info_count_zero", compute_info_count_zero.__doc__, 0x10)


@scenario(host=True)
def compute_info_count_zero_page_boundary():
    """count 0 with the reply offset on a page boundary: the mapping rounds
    to no pages and fails, and the command faults in every setup."""
    return count_zero("compute_info_count_zero_page_boundary", compute_info_count_zero_page_boundary.__doc__, 0)


@scenario
def compute_info_errors():
    """GetComputeInfo faults when: the payload is short; the pipeline ref
    names a buffer ("Failed to get pipeline"), a ref past the list, or an
    entry of type 0; the reply's end lies past the task (in-process
    "Address out of range for task", the service "Failed to make mapping");
    and the reply's count x 8 overflows past the task. Nothing is written
    into the reply buffer."""
    r = Rig("compute_info_errors", compute_info_errors.__doc__)
    reply = r.t.reserve(PAGE)
    buf, _ = r.buffer()
    pso = r.kernel()
    invalid = r.raw(0, b"")
    past = invalid + 1
    r.t.fill(reply, 0x40, 0xee)
    r.ready()
    s = r.s
    s.submit(abi.OP_GET_COMPUTE_INFO, struct.pack("<IIII", r.t.tid, pso, 5, 4), fault=True, label="short payload")
    s.fault_state("short payload")
    s.resume_root()
    for label, ref, off in (("a buffer", buf, reply), ("past the list", past, reply), ("type 0", invalid, reply),
                            ("reply past the task", pso, r.t.length - 8),
                            ("count past the task", pso, reply)):
        count = 0x10000000 if label == "count past the task" else 4
        info(s, r.t.tid, ref, 5, count, off, fault=True, label=label)
        s.fault_state(label)
        s.resume_root()
    r.dump(reply, 0x40, "reply")
    return r.end()


@scenario
def compute_info_no_object_list():
    """A task defined with no object list: mapping the list fails ("Failed to
    map object list" in-process, the service's ensureObjectListMapped), and
    the command faults."""
    from pgtest.scenario import Scenario
    s = Scenario("compute_info_no_object_list", doc=compute_info_no_object_list.__doc__)
    s.bring_up()
    t = GPUTask(s, pages=16)
    t.define()
    info(s, t.tid, 1, 5, 4, 8 * PAGE, fault=True, label="no object list")
    s.fault_state("no object list")
    s.resume_root()
    t.delete()
    s.events("end")
    return s


@scenario(host=True)
def compute_info_ref_zero():
    """Pipeline ref 0 is nil. In-process the replies are sent to nil and
    read 0: the reply is {1, 0} {3, 0} {4, 0} {0, 0}. The service asserts
    the pipeline, so in the isolate setup the service dies, the command
    faults, and the next command on the task faults too."""
    r = Rig("compute_info_ref_zero", compute_info_ref_zero.__doc__, host=True)
    reply = r.t.reserve(PAGE)
    pso = r.kernel()
    r.t.fill(reply, 0x40, 0xee)
    r.ready()
    st = r.s.fifo.cmd(abi.OP_GET_COMPUTE_INFO, struct.pack("<IIIIQ", r.t.tid, 0, 5, 4, reply))
    r.s.doorbell()
    r.s.wait(irqs=r.irqs + 1, label="its interrupt")
    r.s.fault_state("ref 0")
    r.s.resume_root()
    r.s.fifo.wait_stamp(st, label="its stamp, after the resume")
    r.dump(reply, 0x40, "reply")
    info(r.s, r.t.tid, pso, 5, 4, reply + 0x20, timeout_ms=1000, label="a good pipeline next")
    r.s.fault_state("after")
    r.dump(reply, 0x40, "reply")
    r.s.events("end")
    return r.s


@scenario(host=True)
def compute_info_read_only_reply():
    """The reply buffer's page is mapped read-only (page-table bit 31 clear).
    The memory-map setups ask for a writable mapping and fault ("Mapping was
    not writable" or no mapping); the blocks setup maps the task's pages
    through the host and writes the reply anyway."""
    from pgtest.scenario import Scenario
    s = Scenario("compute_info_read_only_reply", host=True, doc=compute_info_read_only_reply.__doc__)
    s.bring_up()
    pages = 64
    t = GPUTask(s, pages=pages, writable=set(range(pages)) - {40})
    reply = 40 * PAGE
    from pgtest.compute_objects import compute_blob, compute_pipeline_record, function_record
    va = t.reserve(0x2000)
    n = s.mem_file(t.gpa(va), "shaders/kernel.metallib")
    fn = t.ol.add(abi.T_FUNCTION, function_record(va, n))
    pso = t.ol.add(abi.T_SERIALIZED, compute_pipeline_record(compute_blob(fn)))
    t.fill(reply, 0x40, 0xee)
    t.define()
    t.install()
    st = s.fifo.cmd(abi.OP_GET_COMPUTE_INFO, struct.pack("<IIIIQ", t.tid, pso, 5, 4, reply))
    s.doorbell()
    s.wait(irqs=3, label="its interrupt")
    s.fault_state("read-only reply")
    s.resume_root()
    s.fifo.wait_stamp(st, label="its stamp, after the resume")
    t.dump(reply, 0x40, "reply")
    t.delete()
    s.events("end")
    return s


@scenario
def compute_info_unknown_task():
    """GetComputeInfo for task 0 (not a user task) and for a task ID never
    defined: DeviceCore::get_user_task_id raises before the assert on the
    task, so the command faults and the assert is unreachable."""
    from pgtest.scenario import Scenario
    s = Scenario("compute_info_unknown_task", doc=compute_info_unknown_task.__doc__)
    s.bring_up()
    for tid in (0, 7):
        info(s, tid, 1, 5, 4, 0, fault=True, label="task %d" % tid)
        s.fault_state("task %d" % tid)
        s.resume_root()
    s.submit(abi.OP_NOP)
    s.events("end")
    return s


@scenario(host=True)
def compute_pipeline_placement():
    """With an object and placement list (opcode 68), the service (argument
    buffers on) writes each compute pipeline's resourceIndex and
    gpuResourceID into its placement entry when it creates it; a guest entry
    with a nonzero resourceID first makes the descriptor force that
    resourceIndex (here 3). A plain pipeline reports index and ID 0; the two
    that support indirect command buffers (blob key 7) report real ones.
    In-process tasks pass no placement and the entries stay as the guest
    wrote them. Each dispatch writes its output."""
    from pgtest.compute_objects import Compute as C
    r = Rig("compute_pipeline_placement", compute_pipeline_placement.__doc__, host=True)
    out, out_va = r.buffer()
    a = r.kernel()
    b = r.kernel(icb=True)
    c = r.kernel(icb=True)
    t = r.t
    t.define()
    n = len(t.ol.entries)
    place = t.reserve(PAGE)
    entries = bytearray(16 * n)
    struct.pack_into("<QQ", entries, 16 * b, 3, 1)
    t.write(place, bytes(entries))
    t.ol.write()
    t.set_object_and_placement_list(0, place, 12 * n, 16 * n)
    r.irqs += 2
    r.compute(C.set_pipeline(a), C.set_buffers(0, [(out, 0)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)),
              C.set_pipeline(b), C.set_buffers(0, [(out, 0x10)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)),
              C.set_pipeline(c), C.set_buffers(0, [(out, 0x20)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)))
    r.dump(out_va, 0x30, "out")
    t.dump(place, 16 * n, "placement list")
    return r.end()
