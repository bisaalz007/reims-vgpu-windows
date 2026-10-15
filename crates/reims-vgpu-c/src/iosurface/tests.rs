use reims_vgpu_core::testkit::FakePlatform;

use super::*;
use crate::device::RvgMemoryRange;
use crate::status::*;
use crate::testkit::{Rec, t_destroy, test_platform};

unsafe extern "C-unwind" fn t_raise(ctx: *mut c_void, vector: u32) {
    Rec::push(ctx, format!("raise {vector}"));
}

fn callbacks(rec: &Arc<Rec>) -> RvgIOSurfaceCallbacks {
    RvgIOSurfaceCallbacks {
        struct_size: sized::size_u32::<RvgIOSurfaceCallbacks>(),
        ctx: rec.ctx(),
        destroy: Some(t_destroy),
        raise_interrupt: Some(t_raise),
    }
}

/// One page of the fake VM's memory.
fn page(p: &FakePlatform) -> RvgMemoryRange {
    let vm = reims_vgpu_core::platform::Platform::vm(p);
    let len = vm.page_size();
    let host = vm.allocate_anywhere(len).unwrap();
    RvgMemoryRange {
        physical_address: 0,
        physical_length: len,
        host_address: host.as_ptr(),
    }
}

/// No memory map is refused with the table's destroy; one is made, answers core's registers,
/// and ends its table at its release
#[test]
fn made_over_a_memory_map() {
    let p = test_platform();
    let rec = Rec::new();
    let cb = callbacks(&rec);
    let mut c = RvgIOSurfaceDeviceConfig {
        struct_size: sized::size_u32::<RvgIOSurfaceDeviceConfig>(),
        memory_map: std::ptr::null(),
        callbacks: &cb,
    };
    let mut d = std::ptr::null_mut();
    // SAFETY: a live config; `d` is writable.
    unsafe {
        assert_eq!(
            rvg_iosurface_device_create(std::ptr::null(), &mut d),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            rvg_iosurface_device_create(&c, &mut d),
            RVG_ERR_NO_MEMORY_MAP
        );
    }
    assert_eq!(rec.destroys().len(), 1);
    let range = page(&p);
    let map = RvgMemoryMap {
        ranges: &range,
        range_count: 1,
    };
    c.memory_map = &map;
    // SAFETY: as above.
    unsafe {
        assert_eq!(
            rvg_iosurface_device_create(&c, std::ptr::null_mut()),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(rvg_iosurface_device_create(&c, &mut d), RVG_OK);
    }
    assert_eq!(rec.destroys().len(), 2);
    // SAFETY: the device made above.
    let core = unsafe { &*d }.device.core();
    for off in (0x1000..0x1038).step_by(8) {
        // SAFETY: as above.
        assert_eq!(
            unsafe { rvg_iosurface_device_mmio_read(d, off) },
            core.mmio_read(off)
        );
    }
    // SAFETY: as above; an offset that names no register is ignored.
    unsafe {
        rvg_iosurface_device_mmio_write(d, 0, 1);
        rvg_iosurface_device_release(d);
        rvg_iosurface_device_release(std::ptr::null_mut());
    }
    assert_eq!(rec.destroys().len(), 3);
}
