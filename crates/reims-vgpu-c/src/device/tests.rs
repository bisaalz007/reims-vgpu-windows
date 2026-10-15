use reims_vgpu_core::guest::records::APV_PAGE_SHIFT;
use reims_vgpu_core::testkit::GuestRam;
use reims_vgpu_core::testkit::host::FakeHost;

use super::*;
use crate::status::*;
use crate::testkit::{HostCtx, Rec, Rig, config, host_table, t_destroy, test_platform};

/// A table with only a destroy, over `rec`.
fn only_destroy(rec: &Arc<Rec>) -> RvgHostCallbacks {
    RvgHostCallbacks {
        struct_size: crate::sized::size_u32::<RvgHostCallbacks>(),
        ctx: rec.ctx(),
        destroy: Some(t_destroy),
        create_task: None,
        destroy_task: None,
        map_memory: None,
        unmap_memory: None,
        read_memory: None,
        raise_interrupt: None,
        add_trace_range: None,
        remove_trace_range: None,
    }
}

fn create(c: &RvgDeviceConfig) -> (RvgStatus, *mut RvgDevice) {
    let mut d = std::ptr::null_mut();
    // SAFETY: a live config; `d` is writable.
    let s = unsafe { rvg_device_create(c, &mut d) };
    (s, d)
}

/// Each refusal in the header's order, with the table's one destroy before the create returns
/// on the caller's thread; a config or table the create cannot read runs none
#[test]
fn refusals_end_the_table() {
    let p = test_platform();
    let rec = Rec::new();
    let table = only_destroy(&rec);
    let mut c = config(1, &table);
    // SAFETY: NULL config.
    let s = unsafe { rvg_device_create(std::ptr::null(), &mut std::ptr::null_mut()) };
    assert_eq!(s, RVG_ERR_INVALID_ARGUMENT);
    c.struct_size = 8;
    assert_eq!(create(&c).0, RVG_ERR_INVALID_ARGUMENT);
    c.struct_size = crate::sized::size_u32::<RvgDeviceConfig>();
    let mut small = table;
    small.struct_size = 8;
    c.callbacks = &small;
    assert_eq!(create(&c).0, RVG_ERR_INVALID_ARGUMENT);
    c.callbacks = &table;
    assert!(rec.destroys().is_empty());

    // SAFETY: a live config; NULL out-pointer.
    let s = unsafe { rvg_device_create(&c, std::ptr::null_mut()) };
    assert_eq!(s, RVG_ERR_INVALID_ARGUMENT);
    c.display_port_count = 0;
    assert_eq!(create(&c).0, RVG_ERR_DISPLAY_PORT_COUNT);
    c.display_port_count = 1;
    c.metal_registry_id = 1;
    assert_eq!(create(&c).0, RVG_ERR_NO_METAL_DEVICE);
    c.metal_registry_id = reims_vgpu_core::testkit::platform::FAKE_REGISTRY_ID;
    c.process_isolation = true;
    assert_eq!(create(&c).0, RVG_ERR_ISOLATION_WITHOUT_MAP);
    c.process_isolation = false;
    let mut with_add = table;
    with_add.add_trace_range = Some(crate::testkit::t_add_trace);
    c.callbacks = &with_add;
    assert_eq!(create(&c).0, RVG_ERR_INVALID_ARGUMENT);
    let null_ranges = RvgMemoryMap {
        ranges: std::ptr::null(),
        range_count: 1,
    };
    c.callbacks = &table;
    c.memory_map = &null_ranges;
    assert_eq!(create(&c).0, RVG_ERR_INVALID_ARGUMENT);
    let me = std::thread::current().id();
    assert_eq!(rec.destroys(), vec![me; 6]);
    drop(p);
}

/// A task or memory callback beside a memory map is a conflict
#[test]
fn memory_conflict() {
    let p = test_platform();
    let host = FakeHost::new(&p, GuestRam::new());
    let ctx = Arc::new(HostCtx {
        rec: Rec::new(),
        blocks: host.blocks(),
    });
    let table = host_table(&ctx);
    let map = RvgMemoryMap {
        ranges: std::ptr::null(),
        range_count: 0,
    };
    let mut c = config(1, &table);
    c.memory_map = &map;
    assert_eq!(create(&c).0, RVG_ERR_MEMORY_CONFLICT);
    assert_eq!(ctx.rec.destroys().len(), 1);
}

/// The table's destroy runs once, after the device's end, on the releasing thread; a longer
/// config with zero past the library's size is accepted
#[test]
fn destroy_after_the_end() {
    let mut rig = Rig::new(1);
    assert!(rig.ctx.rec.destroys().is_empty());
    // SAFETY: the rig's live device.
    let d = unsafe { &*rig.device };
    for off in (0x1000..0x1240).step_by(4) {
        // SAFETY: as above.
        let v = unsafe { rvg_device_mmio_read(rig.device, off) };
        assert_eq!(v, d.device.core().mmio_read(off), "{off:#x}");
    }
    rig.release();
    assert_eq!(rig.ctx.rec.destroys(), vec![std::thread::current().id()]);

    #[repr(C)]
    struct Longer {
        c: RvgDeviceConfig,
        later: u64,
    }
    let rec = Rec::new();
    let table = only_destroy(&rec);
    let mut l = Longer {
        c: config(1, &table),
        later: 0,
    };
    l.c.struct_size = crate::sized::size_u32::<Longer>();
    let (s, d) = create(&l.c);
    assert_eq!(s, RVG_OK);
    assert!(rec.destroys().is_empty());
    // SAFETY: the device made above.
    unsafe { rvg_device_release(d) };
    assert_eq!(rec.destroys().len(), 1);
    l.later = 1;
    assert_eq!(create(&l.c).0, RVG_ERR_UNSUPPORTED);
    assert_eq!(rec.destroys().len(), 1);
}

/// Suspend gives bytes the device takes back; no state is refused with its reason; the
/// status functions refuse NULL
#[test]
fn suspend_and_resume() {
    let rig = Rig::new(1);
    let mut e = 99;
    // SAFETY: the rig's live device; `e` is writable.
    unsafe {
        assert_eq!(
            rvg_device_will_resume(rig.device, std::ptr::null(), 0, &mut e),
            RVG_ERR_RESUME
        );
        assert_eq!(e, RVG_RESUME_ERROR_INVALID_CONTENT);
        rvg_device_will_suspend(rig.device);
        let mut state = RvgBytes {
            data: std::ptr::null_mut(),
            length: 0,
        };
        assert_eq!(rvg_device_finish_suspend(rig.device, &mut state), RVG_OK);
        assert!(!state.data.is_null() && state.length > 0);
        assert_eq!(
            rvg_device_will_resume(rig.device, state.data, state.length, &mut e),
            RVG_OK
        );
        assert_eq!(e, RVG_RESUME_ERROR_NONE);
        assert_eq!(
            rvg_device_will_resume(rig.device, state.data, state.length, std::ptr::null_mut()),
            RVG_OK
        );
        rvg_bytes_free(state);
        rvg_bytes_free(RvgBytes {
            data: std::ptr::null_mut(),
            length: 0,
        });
        rvg_device_did_resume(rig.device);
        rvg_device_pause(rig.device);
        rvg_device_unpause(rig.device);
        rvg_device_reset(rig.device);
        rvg_device_stop(rig.device);
        let mut again = RvgBytes {
            data: std::ptr::null_mut(),
            length: 0,
        };
        assert_eq!(
            rvg_device_finish_suspend(std::ptr::null_mut(), &mut again),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            rvg_device_finish_suspend(rig.device, std::ptr::null_mut()),
            RVG_ERR_INVALID_ARGUMENT
        );
        e = 7;
        assert_eq!(
            rvg_device_will_resume(std::ptr::null_mut(), std::ptr::null(), 0, &mut e),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(e, 7);
        rvg_device_release(std::ptr::null_mut());
    }
}

/// The fake host's callbacks reach the host through the table: the device's order of calls
/// is its twin's, made on the Rust API over the same blocks
#[test]
fn host_calls_are_the_twins() {
    let rig = Rig::new(1);
    // SAFETY: the rig's live device: a reset, which stops and restarts the FIFOs.
    unsafe {
        rvg_device_mmio_write(rig.device, 0x1000, 0);
        rvg_device_reset(rig.device);
    }
    let ours = host_calls(&rig.p);
    let twin_p = std::sync::Arc::new(reims_vgpu_core::testkit::FakePlatform::new());
    let twin_host = FakeHost::new(&twin_p, GuestRam::with_range(0, 0x40_0000).unwrap());
    let b = twin_host.blocks();
    let c = reims_vgpu::DeviceConfig {
        callbacks: reims_vgpu::HostCallbacks {
            create_task: b.create_task,
            destroy_task: b.destroy_task,
            map_memory: b.map_memory,
            unmap_memory: b.unmap_memory,
            read_memory: b.read_memory,
            raise_interrupt: b.raise_interrupt,
            add_trace_range: b.add_trace_range,
            remove_trace_range: b.remove_trace_range,
        },
        ..reims_vgpu::DeviceConfig::default()
    };
    let twin = Device::with_platform(twin_p.clone(), c).unwrap();
    twin.mmio_write(0x1000, 0);
    twin.reset();
    assert_eq!(ours, host_calls(&twin_p));
}

/// The host-block calls `p`'s fake host saw, in order.
fn host_calls(p: &reims_vgpu_core::testkit::FakePlatform) -> Vec<&'static str> {
    p.call_log()
        .calls()
        .into_iter()
        .filter(|c| c.class == "HostBlocks")
        .map(|c| c.sel)
        .collect()
}

/// The device info pairs the guest reads in page `page`, where it asked for them, up to their
/// terminator: 8 bytes each, the key low and the value high.
fn device_info_pairs(rig: &Rig, page: u32) -> Vec<(u32, u32)> {
    // What the device wrote through its mapping of the page, in the guest's RAM.
    rig.p.fake_vm().sync();
    let at = u64::from(page) << APV_PAGE_SHIFT;
    let mut pairs = Vec::new();
    for i in 0..64 {
        let w = rig._host.ram().read_u64(at + 8 * i).unwrap();
        let pair = (
            u32::try_from(w & 0xffff_ffff).unwrap(),
            u32::try_from(w >> 32).unwrap(),
        );
        pairs.push(pair);
        if pair.0 == 0 {
            break;
        }
    }
    pairs
}

/// The feature level reaches the guest: the binary version it reads back after asking for
/// 0xf5, capped by levels 1 to 3; and the device info it reads, which those levels trim (key
/// 18's value, the highest key below 40, and keys 42 and 44); levels 0 and 4 read the same
#[test]
fn the_feature_level_reaches_the_guest() {
    let mut seen = Vec::new();
    let mut infos = Vec::new();
    for level in 0..=4 {
        let rig = Rig::with(1, |c| c.feature_level = level);
        // SAFETY: the rig's live device: the version, then a root node for the mapping.
        let version = unsafe {
            rvg_device_mmio_write(rig.device, 0x1034, 0xf5);
            let v = rvg_device_mmio_read(rig.device, 0x1034);
            rvg_device_mmio_write(rig.device, 0x101c, 1);
            v
        };
        // The guest's device info request (opcode 58) for every key, into guest page 0x20.
        // SAFETY: as above.
        let core = unsafe { &*rig.device }.device.core();
        core.get_device_info(45, 46, 0x20).unwrap();
        let pairs = device_info_pairs(&rig, 0x20);
        let value = |key| pairs.iter().find(|p| p.0 == key).map(|p| p.1);
        let below_40 = pairs.iter().map(|p| p.0).filter(|&k| k < 40).max();
        seen.push((
            version,
            value(18),
            below_40,
            value(42).is_some() && value(44).is_some(),
        ));
        infos.push(pairs);
    }
    assert_eq!(
        seen,
        [
            (0xf5, Some(0x20007), Some(37), true),
            (0x3e, Some(0x20002), Some(18), false),
            (0x71, Some(0x20006), Some(28), false),
            (0xd1, Some(0x20007), Some(37), false),
            (0xf5, Some(0x20007), Some(37), true),
        ]
    );
    assert_eq!(infos[0], infos[4]);
}

/// The first two configs retain their 48/56-byte layouts; sections append at byte 56
#[test]
fn a_first_version_device_config_is_48_bytes() {
    assert_eq!(<RvgDeviceConfig as HostStruct>::V1_SIZE, 48);
    assert_eq!(std::mem::offset_of!(RvgDeviceConfig, memory_files), 48);
    assert_eq!(std::mem::offset_of!(RvgDeviceConfig, memory_sections), 56);
    assert_eq!(size_of::<RvgDeviceConfig>(), 64);
    assert_eq!(size_of::<RvgMemoryRangeSection>(), 32);
    assert_eq!(size_of::<RvgMemoryRangeFile>(), 16);
    assert_eq!(std::mem::offset_of!(RvgMemoryRangeFile, flags), 4);
    assert_eq!(std::mem::offset_of!(RvgMemoryRangeFile, offset), 8);
}

/// `len` bytes of the fake VM's memory, as one range at guest physical 0.
fn fake_range(p: &reims_vgpu_core::testkit::FakePlatform, len: u64) -> RvgMemoryRange {
    let host = reims_vgpu_core::platform::Platform::vm(p)
        .allocate_anywhere(len)
        .unwrap();
    RvgMemoryRange {
        physical_address: 0,
        physical_length: len,
        host_address: host.as_ptr(),
    }
}

/// A host built before `memory_files` passes 48: its config is read without files, whatever
/// follows it in memory; read at its full size, the same bytes are refused, with the table's
/// one destroy
#[test]
fn a_48_byte_device_config_is_accepted() {
    let p = test_platform();
    let rec = Rec::new();
    let table = only_destroy(&rec);
    let range = fake_range(&p, 0x1_0000);
    let map = RvgMemoryMap {
        ranges: &range,
        range_count: 1,
    };
    let bad = RvgMemoryRangeFile {
        fd: -5,
        flags: 1,
        offset: 1,
    };
    let mut c = config(1, &table);
    c.memory_map = &map;
    c.memory_files = &bad;
    assert_eq!(create(&c).0, RVG_ERR_INVALID_ARGUMENT);
    assert_eq!(rec.destroys().len(), 1);
    c.struct_size = 48;
    let (s, d) = create(&c);
    assert_eq!(s, RVG_OK);
    // SAFETY: the device made above.
    unsafe { rvg_device_release(d) };
    assert_eq!(rec.destroys().len(), 2);
}

#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;
