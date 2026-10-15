"""The host as the C API allows it and no other scenario plays it
(docs/testing.md, section 3.1): a NULL error pointer to
rvg_device_will_resume, host callbacks that refuse (create_task returning
NULL; map_memory, unmap_memory and read_memory returning false;
add_trace_range returning NULL) or are left out, and values no other
scenario passes.

reims_vgpu.h makes every callback of RvgHostCallbacks optional, lets
create_task and add_trace_range answer NULL, and has the memory callbacks
answer a bool, so each refusal here is a legal answer. The runner's
descriptor option "refuse" names the calls that fail by their ordinal among
that callback's calls (callbacks come in command order, decision 8, and
every one is in the expected file); a refused call does nothing, and its
record says so.

Setups: a device with a memory map refuses to be created with any task
callback, and never calls them otherwise, so the task-callback scenarios run
in the blocks setup only."""
import struct

from pgtest import abi
from pgtest.display_guest import DisplayGuest, display
from pgtest.efi_guest import EFI
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest import suspend_guest as sg
from pgtest.suspend_guest import blob, state
from pgtest.task import Task

TASK_BLOCKS = ("createTask", "destroyTask", "mapMemory", "unmapMemory", "readMemory")
ALL_IRQS = 0xF  # every raisedInterrupts bit the host sets (display_state.py)


def blocks_but(missing):
    """The blocks setup's task blocks without `missing`."""
    return {b: True for b in TASK_BLOCKS if b != missing}


def efi_on(s, e):
    """An EFI framebuffer present (section 6.22's measured sequence)."""
    e.write_frame(64, 32, 256)
    e.setup()


# ---- 1. rvg_device_will_resume with a NULL error pointer ----

NULL_ERROR_BLOBS = [
    ("empty (the header)", b""),
    ("a PGSD V2 header (the magic)", struct.pack("<IIII", 0x50475344, 2, 22, 1)),
    ("version 2 (the version)", sg.header(version=2)),
    ("an unknown section (the sections)", sg.header() + sg.section(3, b"")),
    ("registers of 76 bytes (a section's length)",
     sg.header() + sg.section(sg.TAG_REGISTERS, sg.registers(state())[:76])),
    ("device info key 43 (the keys a resume reads)", blob(state(), [(43, 0)])),
    ("binary version 0xf6", blob(state(version=0xF6))),
    ("binary version 1 (unsupported)", blob(state(version=1))),
    ("device info key 1 = 0xffffffff (capabilities)", blob(state(), [(1, 0xFFFFFFFF)])),
    ("device info key 13 = 0 (an alignment)", blob(state(), [(13, 0)])),
    ("port count 2", blob(state(ports=2))),
    ("connection types 1", blob(state(connection_types=1))),
]


@scenario
def fresh_host_resume_null_error():
    """rvg_device_will_resume with a NULL error pointer, which the header
    allows, for every kind of refusal: no state, a short, foreign or malformed
    blob (its header, version, sections, a section's length, a device-info
    key), a binary version the device cannot take, a device-info capability or
    alignment the host lacks, the port count and the connection types. Each
    returns RVG_ERR_RESUME and writes nothing through the pointer. A valid blob
    then resumes with a NULL pointer too, and the device runs (the run state a
    failed resume leaves is replaced by the successful one)."""
    s = Scenario("fresh_host_resume_null_error", doc=fresh_host_resume_null_error.__doc__)
    s.api("willResume", nil=True, nil_error=True, label="nil state")
    for label, data in NULL_ERROR_BLOBS:
        s.api("willResume", hex=data.hex(), nil_error=True, label=label)
    s.r(abi.R_VERSION, "after the refusals")
    s.api("willResume", hex=blob(state()).hex(), nil_error=True, label="a valid blob")
    s.api("didResume")
    s.events("resumed")
    s.bring_up(version=None)
    s.submit(abi.OP_NOP)
    s.events("a NOP")
    return s


@scenario
def fresh_host_resume_null_error_roundtrip():
    """A device-written blob resumed on a fresh device with a NULL error
    pointer: the success path never touches it. Then the same blob with
    its port count changed (a device of 2 ports refuses it) and a NULL
    pointer: NO, nothing written."""
    s = Scenario("fresh_host_resume_null_error_roundtrip", displays=[display(0)],
                 doc=fresh_host_resume_null_error_roundtrip.__doc__)
    s.bring_up()
    s.submit(abi.OP_NOP)
    s.events("before the suspend")
    s.api("willSuspend")
    s.api("finishSuspend")
    s.api("recreate")
    s.api("willResume", nil_error=True, label="the device's blob")
    s.api("didResume")
    s.events("resumed")
    s.submit(abi.OP_NOP)
    s.events("a NOP after the resume")
    s.api("recreate", config={"op": "config", "descriptor": {"port_count": 2}}, label="a device of 2 ports")
    s.api("willResume", nil_error=True, label="the blob of a 1-port device")
    return s


# ---- 2. host blocks that refuse ----

@scenario(setups=("blocks",))
def fresh_host_create_task_refused_root():
    """The root task's create_task returns NULL, the header's failure.
    DeviceCore::set_root_page then cannot map the root node and calls
    destroy_task with that NULL, as the header says it may after a failed root
    map: the runner's host records the call as given and does nothing. The
    register keeps the page. Writing the root page again creates a task and
    maps the node, and the device then works."""
    s = Scenario("fresh_host_create_task_refused_root", setups=("blocks",),
                 descriptor={"refuse": {"createTask": [1]}}, doc=fresh_host_create_task_refused_root.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    s.events("root page: its task refused")
    s.r(abi.R_ROOT_PAGE, "root page register")
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    s.events("root page again")
    s.bring_up(version=None, root=False)
    s.submit(abi.OP_NOP)
    s.events("a NOP")
    s.diff("after")
    return s


@scenario(setups=("blocks",))
def fresh_host_create_task_refused_root_fifo():
    """The root task refused (as fresh_host_create_task_refused_root), and the
    guest goes on without writing the root page again: the root FIFO's ring
    cannot be set up without a root node (DeviceCore::create_root_fifo refuses
    when there is none): no callback is called, the enable reads back 0, and
    the doorbell's NOP never completes. No fault is raised and no interrupt
    comes."""
    s = Scenario("fresh_host_create_task_refused_root_fifo", setups=("blocks",), watchdog_s=15,
                 descriptor={"refuse": {"createTask": [1]}}, doc=fresh_host_create_task_refused_root_fifo.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    s.events("root page: its task refused")
    s.bring_up(version=None, root=False)
    s.submit(abi.OP_NOP, timeout_ms=500)
    s.events("a NOP")
    s.fault_state("after")
    s.r(abi.R_FIFO_ENABLE, "enable")
    s.diff("after")
    return s


@scenario(setups=("blocks",))
def fresh_host_create_task_refused_define():
    """A task define whose create_task returns NULL: the in-process task's
    creation fails and DeviceCore::create_task_id raises (no task was made), a
    fault at the define. After the resume a NOP runs, and the same define again
    (a create_task the host grants) creates the task, which maps."""
    s = Scenario("fresh_host_create_task_refused_define", setups=("blocks",),
                 descriptor={"refuse": {"createTask": [2]}}, doc=fresh_host_create_task_refused_define.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=4)
    t.define(fault=True)
    s.events("the define: its task refused")
    s.fault_state("after the define")
    s.resume_root()
    s.submit(abi.OP_NOP)
    t.define()
    t.map()
    s.events("a NOP, the define again, a map")
    s.diff("after")
    return s


@scenario(setups=("blocks",))
def fresh_host_create_task_refused_efi():
    """The EFI framebuffer's task refused: the task rebuild
    (EfiDisplay::rebuild_task) logs it and leaves the task not ready, so
    nothing is presented and no callback comes. Turning EFI off and on again
    creates the task (the next create_task is granted) and presents."""
    s = Scenario("fresh_host_create_task_refused_efi", setups=("blocks",), displays=[display(0)],
                 descriptor={"refuse": {"createTask": [1]}}, doc=fresh_host_create_task_refused_efi.__doc__)
    e = EFI(s)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    efi_on(s, e)
    s.wait(cb="newFrame", timeout_ms=500, label="no frame")
    s.events("EFI on: its task refused", collapse=["newFrame"])
    e.regs("EFI registers")
    s.w(abi.R_EFI_ENABLE, 0)
    s.events("EFI off", collapse=["newFrame"])
    efi_on(s, e)
    s.wait(cb="newFrame", label="a frame")
    s.events("EFI on again", collapse=["newFrame"])
    s.encode(0, samples=[(0, 0), (63, 31)], label="the frame")
    s.w(abi.R_EFI_ENABLE, 0)
    s.events("EFI off", collapse=["newFrame"])
    return s


@scenario(setups=("blocks",))
def fresh_host_map_memory_refused_root():
    """The root node's map_memory returns false: DeviceCore::set_root_page
    calls destroy_task on the root task but keeps its handle (only its base is
    cleared), and leaves no root node. A root FIFO set up then calls no block
    (there is no root node to hang it on): the enable reads back 0 and the NOP
    never completes, with no fault. Writing the root page again creates a new
    task (the kept, destroyed one is simply replaced, never destroyed again)
    and the device works."""
    s = Scenario("fresh_host_map_memory_refused_root", setups=("blocks",), watchdog_s=15,
                 descriptor={"refuse": {"mapMemory": [1]}}, doc=fresh_host_map_memory_refused_root.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    s.events("root page: its mapping refused")
    s.bring_up(version=None, root=False)
    s.submit(abi.OP_NOP, timeout_ms=500)
    s.events("a root FIFO without a root node")
    s.fault_state("after")
    s.r(abi.R_FIFO_ENABLE, "enable")
    s.w(abi.R_FIFO_ENABLE, 0)
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    s.events("root page again")
    s.bring_up(version=None, root=False)
    s.submit(abi.OP_NOP)
    s.events("a NOP")
    s.diff("after")
    return s


@scenario(setups=("blocks",))
def fresh_host_map_memory_refused_fifo():
    """The root FIFO's ring mapping refused (the second map_memory): the enable
    is taken, and nothing the guest rings runs. No fault is raised, no
    interrupt comes, and a define and a NOP behind it never complete: a silent
    stall (the ring's mapping, DeviceCore::new_contiguous_mapping, answers
    none, so DeviceCore::create_root_fifo makes no FIFO)."""
    s = Scenario("fresh_host_map_memory_refused_fifo", setups=("blocks",), watchdog_s=15,
                 descriptor={"refuse": {"mapMemory": [2]}}, doc=fresh_host_map_memory_refused_fifo.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=4)
    t.define(timeout_ms=500)
    s.submit(abi.OP_NOP, timeout_ms=500)
    s.events("a define and a NOP")
    s.fault_state("after")
    s.r(abi.R_FIFO_READ, "read")
    s.diff("after")
    return s


@scenario(setups=("blocks",))
def fresh_host_map_memory_refused_task():
    """MapMemory2's map_memory refused for the task's first range (the fourth
    map_memory: root node, FIFO ring, task root page, then the task's).
    DeviceCore::map_memory_in_task answers false, LocalTask::map_into_host
    raises and the map faults; the second range is never asked for. After the
    resume the same map is granted."""
    s = Scenario("fresh_host_map_memory_refused_task", setups=("blocks",),
                 descriptor={"refuse": {"mapMemory": [4]}}, doc=fresh_host_map_memory_refused_task.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=4)
    t.define()
    s.events("defined")
    t.map(fault="either")
    s.events("the map: refused")
    s.resume_root()
    t.map()
    s.events("the map again")
    t.write(0, b"task data")
    t.delete()
    s.events("deleted")
    s.diff("after")
    return s


@scenario(setups=("blocks",))
def fresh_host_unmap_memory_refused():
    """unmap_memory returns false, the header's failure (QEMU gives a memory
    map and no unmap_memory): DeviceCore::unmap_memory_in_task answers false
    and LocalTask::unmap_memory raises, a fault at the UnmapMemory (offset
    0x3c). After the resume the same unmap asks the host again and faults again
    (offset 0x5c). Deleting the task completes: the release of the task's
    root-page mapping (a Mapping over the host's task) ignores the refused
    unmap, and destroy_task follows."""
    s = Scenario("fresh_host_unmap_memory_refused", setups=("blocks",),
                 descriptor={"refuse": {"unmapMemory": True}}, doc=fresh_host_unmap_memory_refused.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=4)
    t.define()
    t.map()
    s.events("mapped")
    t.unmap(fault=True)
    s.events("the unmap: refused")
    s.fault_state("after the unmap")
    s.resume_root()
    t.unmap(fault="either")
    s.events("the unmap again")
    s.fault_state("after the second unmap")
    s.resume_root()
    t.delete(fault="either")
    s.events("deleted")
    s.diff("after")
    return s


@scenario(setups=("blocks",))
def fresh_host_read_memory_refused():
    """read_memory returns false for every read (the header's failure):
    a define reads nothing, and completes; MapMemory2's page-table walk
    reads the task's table through it and faults at the map. Nothing is
    mapped."""
    s = Scenario("fresh_host_read_memory_refused", setups=("blocks",),
                 descriptor={"refuse": {"readMemory": True}}, doc=fresh_host_read_memory_refused.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=4)
    t.define()
    s.events("defined")
    t.map(fault=True)
    s.events("the map: its reads refused")
    s.fault_state("after")
    s.diff("after")
    return s


@scenario
def fresh_host_trace_refused():
    """add_trace_range returns NULL (the header's "none") for the EFI
    framebuffer: the framebuffer presents anyway, and turning EFI off calls no
    remove_trace_range for the NULL range. Its guest writes cannot be
    reported, so no trace_fire runs."""
    s = Scenario("fresh_host_trace_refused", displays=[display(0)],
                 descriptor={"trace": True, "refuse": {"addTraceRange": True}}, doc=fresh_host_trace_refused.__doc__)
    e = EFI(s)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    efi_on(s, e)
    s.wait(cb="newFrame", label="a frame")
    s.events("EFI on: its trace range refused", collapse=["newFrame"])
    s.encode(0, samples=[(0, 0), (63, 31)], label="the frame")
    s.w(abi.R_EFI_ENABLE, 0)
    s.events("EFI off", collapse=["newFrame"])
    return s


# ---- 2. host blocks left out ----

@scenario(setups=("blocks",))
def fresh_host_trace_remove_only():
    """A host with remove_trace_range and no add_trace_range (the header:
    without add_trace_range the device calls neither; only the reverse is
    refused): the EFI framebuffer presents, and no trace callback is
    called."""
    s = Scenario("fresh_host_trace_remove_only", setups=("blocks",), displays=[display(0)],
                 descriptor={"trace": {"remove": True}}, doc=fresh_host_trace_remove_only.__doc__)
    e = EFI(s)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    efi_on(s, e)
    s.wait(cb="newFrame", label="a frame")
    s.events("EFI on", collapse=["newFrame"])
    s.w(abi.R_EFI_ENABLE, 0)
    s.events("EFI off", collapse=["newFrame"])
    return s


@scenario(setups=("blocks",))
def fresh_host_no_destroy_task_reset_dies():
    """A blocks host without destroy_task (every callback is optional at
    creation): the device comes up and runs, and DeviceCore::reset tears the
    root down: DeviceCore::teardown_root finds no destroy_task callback and
    ends the process. The host dies in the reset."""
    s = Scenario("fresh_host_no_destroy_task_reset_dies", setups=("blocks",),
                 descriptor={"blocks": blocks_but("destroyTask")}, doc=fresh_host_no_destroy_task_reset_dies.__doc__)
    s.bring_up()
    s.submit(abi.OP_NOP)
    s.events("up")
    s.api("reset")
    return s


@scenario(setups=("blocks",))
def fresh_host_no_destroy_task_root_page_dies():
    """As fresh_host_no_destroy_task_reset_dies, but the guest writes a new
    root page: DeviceCore::set_root_page tears the old root down first, and
    DeviceCore::teardown_root finds no destroy_task callback and ends the
    process in the MMIO write."""
    s = Scenario("fresh_host_no_destroy_task_root_page_dies", setups=("blocks",),
                 descriptor={"blocks": blocks_but("destroyTask")},
                 doc=fresh_host_no_destroy_task_root_page_dies.__doc__)
    s.bring_up()
    s.submit(abi.OP_NOP)
    s.events("up")
    s.w(abi.R_ROOT_PAGE, s.pfn(s.alloc_page()))
    return s


@scenario(setups=("blocks",))
def fresh_host_no_destroy_task_delete_dies():
    """A blocks host without destroy_task deletes a task:
    DeviceCore::destroy_task finds no destroy_task callback and ends the
    process on the FIFO thread, after the DeleteTask's doorbell."""
    s = Scenario("fresh_host_no_destroy_task_delete_dies", setups=("blocks",),
                 descriptor={"blocks": blocks_but("destroyTask")}, doc=fresh_host_no_destroy_task_delete_dies.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=4)
    t.define()
    t.map()
    s.events("a task")
    t.delete()
    return s


@scenario(setups=("blocks",))
def fresh_host_no_create_task_dies():
    """A blocks host without create_task: the root page write
    (DeviceCore::set_root_page) finds no create_task callback and ends the
    process in the MMIO write."""
    s = Scenario("fresh_host_no_create_task_dies", setups=("blocks",),
                 descriptor={"blocks": blocks_but("createTask")}, doc=fresh_host_no_create_task_dies.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    return s


@scenario(setups=("blocks",))
def fresh_host_no_map_memory_dies():
    """A blocks host without map_memory: the root node's mapping finds no
    map_memory callback and ends the process in the root page write."""
    s = Scenario("fresh_host_no_map_memory_dies", setups=("blocks",),
                 descriptor={"blocks": blocks_but("mapMemory")}, doc=fresh_host_no_map_memory_dies.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    return s


@scenario(setups=("blocks",))
def fresh_host_no_unmap_memory_dies():
    """A blocks host without unmap_memory runs until something unmaps: an
    UnmapMemory (DeviceCore::unmap_memory_in_task) finds no unmap_memory
    callback and ends the process on the FIFO thread."""
    s = Scenario("fresh_host_no_unmap_memory_dies", setups=("blocks",),
                 descriptor={"blocks": blocks_but("unmapMemory")}, doc=fresh_host_no_unmap_memory_dies.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=4)
    t.define()
    t.map()
    s.events("a task")
    t.unmap()
    return s


# ---- 3. values no other scenario passes ----

@scenario
def fresh_host_display_mode_values():
    """RvgDisplayMode takes any double refresh rate, and the display
    descriptor any size in millimetres. The mode list holds the refresh
    rate as (uint32_t)floor(rate * 65536 + 0.5), an arm64 conversion that
    saturates: 0 and anything negative, NaN and -inf give 0; 65536 and
    above, and inf, give 0xffffffff. The sizes are stored as floats as
    given, NaN and negative included. Each set_modes bumps the list's
    counter and interrupts whether or not the values changed."""
    s = Scenario("fresh_host_display_mode_values", displays=[display(0, mm=["nan", -5])],
                 doc=fresh_host_display_mode_values.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0, enabled=ALL_IRQS)
    g.ack(0, 1)
    ack = 1
    for label, rate in (("refresh 0", 0), ("refresh -60", -60), ("refresh 65536", 65536), ("refresh 1e300", 1e300),
                        ("refresh nan", "nan"), ("refresh inf", "inf"), ("refresh -inf", "-inf"),
                        ("refresh 65535.99999", 65535.99999), ("refresh 65535.99", 65535.99)):
        s.api("set_modes", display=0, modes=[[64, 32, rate]], label=label)
        s.quiesce()
        s.events(label)
        g.diff_page(0, label)
        ack += 1
        g.ack(0, ack, fault=False)
    s.api("display_props", display=0)
    s.events("end")
    return s


@scenario
def fresh_host_iosfc_second_refused():
    """A second IOSurface host device in the process, without a memory map:
    rvg_iosurface_device_create refuses it (RVG_ERR_NO_MEMORY_MAP), and the
    first device keeps answering."""
    s = Scenario("fresh_host_iosfc_second_refused", iosfc=True, doc=fresh_host_iosfc_second_refused.__doc__)
    s.iosfc_r(0, "register 0")
    s.api("iosfc_again", no_map=True)
    s.iosfc_r(0, "register 0 after")
    return s
