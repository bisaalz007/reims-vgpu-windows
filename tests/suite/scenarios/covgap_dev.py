"""The device, its FIFOs, the displays, the IOSurface host device and the
decoders: the few paths a hypervisor can reach there that the rest of the
suite leaves unreached.

Most of what the rest of the suite leaves unreached there is not reachable:
private descriptor properties and private methods QEMU never declares,
allocation failures, Metal calls that never fail on this host, races only
timing opens, and code nothing calls. What is left, each here: a barrier
wait on a child FIFO whose stamp index the guest lowered the version below,
a page-table read through a root-task mapping that cannot be made, an
IOSurface host device whose memory map is refused, and a display pipeline
Metal cannot make. Two more lock why a path stays unreached on this host:
device-info key 36 at the host's own maximum and compute dispatch types
Metal accepts unchecked.
"""
from pgtest import abi
from pgtest.abi import PAGE
from pgtest.display_guest import DisplayGuest, display, p_swap
from pgtest.fifo_util import child
from pgtest.registry import scenario
from pgtest.scenario import RAM_BASE, RAM_SIZE, Scenario
from pgtest.suspend_guest import blob, state
from pgtest.task import Task, seg
from pgtest.compute_objects import Blit as B
from pgtest.tasks_lib import fault_and_resume


@scenario
def covgap_dev_barrier_waiter_above_count_dies():
    """A barrier wait records the waiting FIFO's stamp index, and
    DeviceCore::add_wait asserts it is below maxFIFOCount, which the version
    sets (32 below 61, 64 from it) while a child's stamp index is checked
    against it only at its creation. Children 1 and 2, with stamp indices 31
    and 32, are created at version 245; the guest then writes version 60. Child
    1 (31) blocks on root slot 0 (a 300 ms wait that times out is the
    behaviour) and a root NOP releases it. Child 2 (32) then waits on the root
    slot too: the assert fails (32 < 32 is false) and the host dies at the
    wait."""
    s = Scenario("covgap_dev_barrier_waiter_above_count_dies", stamp_slots=33,
                 doc=covgap_dev_barrier_waiter_above_count_dies.__doc__)
    s.bring_up()
    a = child(s, 1, stamp_index=31)
    b = child(s, 2, stamp_index=32)
    s.w(abi.R_VERSION, 60)
    st = a.cmd(abi.OP_NOP, barriers=[(0, s.fifo.stamp + 1)])
    a.doorbell()
    a.wait_stamp(st, timeout_ms=300, label="child 1 (stamp index 31) blocked on 0:next (times out)")
    s.submit(abi.OP_NOP, label="the root NOP child 1 waits for")
    a.wait_stamp(st, label="child 1 released")
    s.events("index 31 at version 60")
    st = b.cmd(abi.OP_NOP, barriers=[(0, s.fifo.stamp + 1)])
    b.doorbell()
    b.wait_stamp(st, label="child 2 (stamp index 32) waits: the death")
    s.events("not reached")
    return s


@scenario(host=True)
def covgap_dev_resume_nesting_depth_above_host():
    """device_info::check_capabilities fails a device-info value the host has
    less of (the comparison is `<`, so an equal value passes): key 36
    (maxPredicatedNestingDepth), which the host writes from version 202
    (supportsCommandBufferJump). This host writes 0xffffffff
    (device_info_keys), so 0xffffffff and 0x80000000 at 245 and 202, and 1 at
    245, are all accepted: the refusal needs a host with a smaller depth.
    Nothing is resumed. Keyed by the host."""
    s = Scenario("covgap_dev_resume_nesting_depth_above_host", host=True,
                 doc=covgap_dev_resume_nesting_depth_above_host.__doc__)
    for version, v in ((245, 0xFFFFFFFF), (202, 0x80000000), (245, 1)):
        s.api("willResume", hex=blob(state(version=version), [(36, v)]).hex(),
              label="version %d, key 36 = %#x" % (version, v))
    s.r(abi.R_VERSION, "after the last")
    return s


@scenario(setups=("blocks",))
def covgap_dev_no_read_memory_table_outside_ram():
    """The blocks setup without read_memory reads a task's page table
    through a one-page root-task mapping; when the table's page is outside
    guest RAM the host's map_memory fails, so the mapping is nil and the
    read fails. Task 1's page-table root names a page 256 MiB past RAM: its
    define succeeds (it reads no table), and its map faults after the
    host's map_memory of that page into the root task answers false; a task
    2 with a table in RAM then maps. Blocks setup only: the map and isolate
    setups read through the memory map."""
    s = Scenario("covgap_dev_no_read_memory_table_outside_ram", setups=("blocks",),
                 descriptor={"blocks": {"createTask": True, "destroyTask": True, "mapMemory": True,
                                        "unmapMemory": True}},
                 doc=covgap_dev_no_read_memory_table_outside_ram.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=4)
    outside = s.pfn(0) + (0x10000000 >> 14)
    t.write_root(pt_root=outside)
    s.events("before")
    t.define()
    s.events("define: the table is not read")
    t.map(fault=True)
    fault_and_resume(s, "map, table outside RAM")
    s.events("map")
    u = Task(s, tid=2, pages=4)
    u.define()
    u.map()
    s.events("task 2 defined and mapped")
    u.dump(0, 16, label="task 2's first bytes")
    return s


@scenario
def covgap_dev_iosfc_map_refused():
    """An IOSurface host device whose memory map is refused (two guest RAM
    ranges with the same start): rvg_iosurface_device_create refuses it
    (RVG_ERR_MEMORY_MAP). The RvgDevice, given only the first range where it
    has a map, is created with use_iosurface_mapper and no mapper service, so
    opcode 8 is "Invalid mappingID" with its stamp signalled and nothing
    presented. Every setup: the runner's IOSurface host device always gets a
    memory map of all guest RAM."""
    s = Scenario("covgap_dev_iosfc_map_refused", iosfc=True, displays=[display(0)],
                 ram=[(RAM_BASE, RAM_SIZE), (RAM_BASE, PAGE)],
                 descriptor={"map_ranges": [[hex(RAM_BASE), hex(RAM_SIZE), hex(RAM_BASE)]]},
                 doc=covgap_dev_iosfc_map_refused.__doc__)
    g = DisplayGuest(s)
    g.connect(0, enabled=0xF)
    g.ack(0, 1)
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="mapping 1")
    s.fault_state("after")
    s.events("presents")
    g.diff_page(0, "page")
    return s



@scenario(host=True)
def covgap_dev_encode_integer_target_dies():
    """RGBA8Uint (73) is not a frame target format: rvg_frame_target_create
    refuses it (RVG_ERR_INVALID_ARGUMENT) before any present and after one,
    and the scenario runs on to its end: nothing dies. Between them, after a
    present, BGRA8Unorm encodes the frame. QEMU always encodes into
    BGRA8Unorm. Keyed by the host: the display's shaders draw the pixels."""
    from pgtest.display_guest import p_present2, pattern
    s = Scenario("covgap_dev_encode_integer_target_dies", host=True, displays=[display(0)],
                 doc=covgap_dev_encode_integer_target_dies.__doc__)
    g = DisplayGuest(s, pages=128)
    ref, _ = g.backing(16, 8, pixels=pattern(16, 8, 5))
    g.install()
    g.connect(0, enabled=0xF)
    g.ack(0, 1)
    s.encode(0, 16, 8, label="RGBA8Uint, no frame", format=73, samples=[[0, 0]])
    s.submit(abi.OP_DISPLAY_TRANSACTION2, p_present2(0, ref), label="present")
    s.wait(cb="newFrame")
    s.encode(0, 16, 8, label="BGRA8Unorm", samples=[[0, 0], [15, 7]])
    s.encode(0, 16, 8, label="RGBA8Uint, after a frame", format=73, samples=[[0, 0]])
    s.events("after both refusals")
    return s



@scenario
def covgap_dev_compute_dispatch_type_unknown():
    """SetComputePassDispatchType (0xdb) as a compute segment's first record
    sets the pass descriptor's dispatchType to the guest's u32 unchecked,
    and Metal makes the encoder for any value, so the decoder's "Failed to
    create compute encoder" is not reached: types 2 and 0xffffffff each
    dispatch the kernel (out[i] = 2i) and the exec completes."""
    from pgtest.compute_objects import Compute as C, Rig
    r = Rig("covgap_dev_compute_dispatch_type_unknown", covgap_dev_compute_dispatch_type_unknown.__doc__)
    out, out_va = r.buffer()
    pso = r.kernel()
    r.ready()
    for t in (2, 0xFFFFFFFF):
        r.run(seg(abi.ENC_BLIT, B.fill(out, 0, 0x10, 0xEE)))
        r.compute(C.dispatch_type(t), C.set_pipeline(pso), C.set_buffers(0, [(out, 0)]),
                  C.dispatch_threads((4, 1, 1), (4, 1, 1)), label="type %#x" % t)
        r.dump(out_va, 0x10, "out, type %#x" % t)
    return r.end()
