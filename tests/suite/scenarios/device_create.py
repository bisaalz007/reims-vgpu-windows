"""Device creation through rvg_device_create's config (docs/testing.md,
sections 3.1 and 3.3).

rvg_device_create refuses a display port count of 0 or above
RVG_MAX_DISPLAY_PORTS (8), a Metal registry ID no device has, a memory map
together with any task callback, a memory map that is refused (overlapping
ranges, an address plus length that overflows), add_trace_range without
remove_trace_range, and isolation without a memory map, each with its
status. The config's defaults: one display port (the runner's), feature
level 0 and argument buffers on.
"""
from pgtest import abi
from pgtest.registry import scenario
from pgtest.scenario import RAM_BASE, RAM_SIZE, Scenario

BLOCKS = ("createTask", "destroyTask", "mapMemory", "unmapMemory", "readMemory")


def nop_check(s):
    """A device that was created runs a NOP (section 6.25 a)."""
    s.bring_up()
    s.submit(abi.OP_NOP)
    s.events("a NOP")
    s.sweep(0x1000, 0x1300, label="running")


@scenario
def device_create_port_count():
    """display_port_count 0 and 9 refuse the device
    (RVG_ERR_DISPLAY_PORT_COUNT); 1 (the runner's default) to 8 create one,
    and 0x122c reads the count at version 245."""
    out = []
    for n in (0, 9, 0x80000000, 1, 3, 8):
        name = "device_create_port_count_%x_new" % n
        s = Scenario(name, descriptor={"port_count": n}, doc=device_create_port_count.__doc__)
        s.r(abi.R_PORT_COUNT, "before a version")
        s.w(abi.R_VERSION, abi.MAX_VERSION)
        s.r(abi.R_PORT_COUNT, "at 245")
        out.append(s)
    return out


@scenario
def device_create_no_metal_device():
    """A config whose metal_registry_id names no Metal device (the runner
    passes UINT64_MAX) is refused: RVG_ERR_NO_METAL_DEVICE."""
    return Scenario("device_create_no_metal_device_new", descriptor={"no_mtl_device": True},
                    doc=device_create_no_metal_device.__doc__)


@scenario(setups=("map", "isolate"))
def device_create_map_with_a_block():
    """A memory map together with any one of the five task blocks is
    refused. Map and isolate setups only: they are the ones with a memory
    map."""
    out = []
    for b in BLOCKS:
        out.append(Scenario("device_create_map_with_" + b, setups=("map", "isolate"),
                            descriptor={"blocks": {b: True}}, doc=device_create_map_with_a_block.__doc__))
    return out


@scenario(setups=("blocks",))
def device_create_blocks_with_map():
    """The blocks setup with a memory map as well is refused. Blocks
    setup only: it is the one with blocks."""
    return Scenario("device_create_blocks_with_map", setups=("blocks",), descriptor={"map_too": True},
                    doc=device_create_blocks_with_map.__doc__)


@scenario(setups=("map", "isolate"))
def device_create_map_ranges():
    """Memory maps the device refuses or accepts: two overlapping ranges,
    and a range whose address plus length overflows, are refused; two
    adjacent ranges, ranges given out of order, and a zero-length range are
    accepted, and a NOP runs over them. Map and isolate setups only: the
    blocks setup has no map."""
    half = RAM_SIZE // 2
    cases = {
        "overlap": [[RAM_BASE, half + 0x4000, RAM_BASE], [RAM_BASE + half, half, RAM_BASE + half]],
        "same_start": [[RAM_BASE, 0x4000, RAM_BASE], [RAM_BASE, 0x4000, RAM_BASE]],
        "overflow": [[RAM_BASE, RAM_SIZE, RAM_BASE], [0xFFFFFFFFFFFFC000, 0x8000, RAM_BASE]],
        "overflow_end": [[0xFFFFFFFFFFFFC000, 0x4000, RAM_BASE], [RAM_BASE, RAM_SIZE, RAM_BASE]],
        "adjacent": [[RAM_BASE, half, RAM_BASE], [RAM_BASE + half, half, RAM_BASE + half]],
        "reversed": [[RAM_BASE + half, half, RAM_BASE + half], [RAM_BASE, half, RAM_BASE]],
        "zero_length": [[RAM_BASE, RAM_SIZE, RAM_BASE], [0x80000000, 0, RAM_BASE]],
    }
    out = []
    for k, ranges in cases.items():
        s = Scenario("device_create_map_" + k, setups=("map", "isolate"),
                     descriptor={"map_ranges": [[hex(a), hex(n), hex(h)] for a, n, h in ranges]},
                     doc=device_create_map_ranges.__doc__)
        nop_check(s)
        out.append(s)
    return out


@scenario(setups=("isolate",))
def device_create_isolation_without_map():
    """Isolation without a memory map (and without blocks) is refused.
    Isolate setup only: the refusal needs isolation on."""
    return Scenario("device_create_isolation_without_map", setups=("isolate",), descriptor={"no_map": True},
                    doc=device_create_isolation_without_map.__doc__)


@scenario(setups=("map",))
def device_create_nothing_dies_on_root_page():
    """No task callbacks and no memory map: the device is created, its
    registers work, and the first nonzero root-page write finds no
    create_task callback: the host dies. Map setup only: the runner installs
    no task callbacks there, and without a map it has nothing else."""
    s = Scenario("device_create_nothing_dies_on_root_page", setups=("map",), descriptor={"no_map": True},
                 doc=device_create_nothing_dies_on_root_page.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    s.sweep(0x1000, 0x1300, label="no blocks, no map")
    s.w(abi.R_ROOT_PAGE, 0)
    s.r(abi.R_ROOT_PAGE, "root page 0 is harmless")
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    return s


@scenario
def device_create_no_raise_interrupt():
    """Without a raise_interrupt callback the device is created and runs,
    and the first interrupt ends the process. The first interrupt
    here is -unpause's, which it raises on the calling thread, so the op it
    dies at is certain; a stamp's would be raised on the FIFO's thread,
    racing the runner's ops."""
    s = Scenario("device_create_no_raise_interrupt", descriptor={"no_raise": True},
                 doc=device_create_no_raise_interrupt.__doc__)
    s.bring_up()
    s.sweep(0x1000, 0x1300, label="running without raiseInterrupt")
    s.api("pause")
    s.api("unpause")
    return s


@scenario
def device_create_trace_blocks():
    """add_trace_range without remove_trace_range is refused at creation
    (RVG_ERR_INVALID_ARGUMENT). remove_trace_range alone is ignored, and
    both together create a device."""
    out = []
    for k, which in (("add_only", {"add": True}), ("remove_only", {"remove": True}), ("both", True)):
        s = Scenario("device_create_trace_" + k, descriptor={"trace": which}, doc=device_create_trace_blocks.__doc__)
        if k != "add_only":
            nop_check(s)
        out.append(s)
    return out


@scenario(setups=("blocks",))
def device_create_partial_blocks():
    """The blocks setup with some task callbacks left out: every one is
    optional at creation. Without read_memory, a task's page-table walk
    reads guest memory through a root-task mapping instead: map_memory and
    unmap_memory calls on the root task take its place. Blocks setup
    only."""
    s = Scenario("device_create_no_read_memory", setups=("blocks",),
                 descriptor={"blocks": {"createTask": True, "destroyTask": True, "mapMemory": True,
                                        "unmapMemory": True}},
                 doc=device_create_partial_blocks.__doc__)
    from pgtest.task import Task
    s.bring_up()
    t = Task(s, tid=1, pages=4)
    t.define()
    t.map()
    s.events("a task defined and mapped")
    t.delete()
    s.events("deleted")
    s.diff("after")
    return s


@scenario(setups=("map", "isolate"))
def device_create_iosurface_mapper():
    """The config QEMU's apple-gfx-mmio gives: use_iosurface_mapper set and
    argument buffers on, with an IOSurface host device created first, and
    after. The device runs and its device info is the setup's. Map and
    isolate setups only: the IOSurface host device needs a memory map."""
    out = []
    for after in (False, True):
        name = "device_create_iosurface_mapper" + ("_after" if after else "")
        s = Scenario(name, setups=("map", "isolate"), iosfc=True, iosfc_after=after, host=True,
                     doc=device_create_iosurface_mapper.__doc__)
        nop_check(s)
        import struct
        page = s.alloc_page()
        s.submit(abi.OP_GET_DEVICE_INFO, struct.pack("<III", 64, 64, s.pfn(page)))
        s.dump(s.pa(page), 0x180, "device info")
        out.append(s)
    return out
