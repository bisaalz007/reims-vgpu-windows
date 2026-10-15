use reims_vgpu_core::testkit::CompleteOn;

use super::*;
use crate::display::{RvgDisplayConfig, rvg_display_create, rvg_display_release};
use crate::status::*;
use crate::testkit::Rig;

fn target(device: *mut RvgDevice, w: u32, h: u32, format: u32) -> (RvgStatus, *mut RvgFrameTarget) {
    let c = RvgFrameTargetConfig {
        struct_size: sized::size_u32::<RvgFrameTargetConfig>(),
        width: w,
        height: h,
        format,
    };
    let mut t = std::ptr::null_mut();
    // SAFETY: a live device or NULL, a live config; `t` is writable.
    let s = unsafe { rvg_frame_target_create(device, &c, &mut t) };
    (s, t)
}

fn display(device: *mut RvgDevice) -> *mut RvgDisplay {
    let c = RvgDisplayConfig {
        struct_size: sized::size_u32::<RvgDisplayConfig>(),
        serial_number: 0x5eed_f001,
        port: 0,
        name: std::ptr::null(),
        name_length: 0,
        width_millimeters: 0.0,
        height_millimeters: 0.0,
        callbacks: std::ptr::null(),
        window: std::ptr::null(),
    };
    let mut d = std::ptr::null_mut();
    // SAFETY: a live device and config; `d` is writable.
    assert_eq!(unsafe { rvg_display_create(device, &c, &mut d) }, RVG_OK);
    d
}

/// Sizes 1 to 16384 and the four formats; anything else is an invalid argument
#[test]
fn target_refusals() {
    let rig = Rig::new(1);
    for (w, h, f) in [
        (0, 1, RVG_PIXEL_FORMAT_BGRA8_UNORM),
        (1, 16385, RVG_PIXEL_FORMAT_BGRA8_UNORM),
        (1, 1, 73),
        (1, 1, 0),
    ] {
        assert_eq!(
            target(rig.device, w, h, f).0,
            RVG_ERR_INVALID_ARGUMENT,
            "{w} {h} {f}"
        );
    }
    assert_eq!(
        target(std::ptr::null_mut(), 1, 1, 80).0,
        RVG_ERR_INVALID_ARGUMENT
    );
    for f in [70, 71, 80, 81] {
        let (s, t) = target(rig.device, 16384, 1, f);
        assert_eq!(s, RVG_OK);
        // SAFETY: the target made above.
        unsafe {
            assert_eq!(rvg_frame_target_retain(t), t);
            rvg_frame_target_release(t);
            rvg_frame_target_release(t);
        }
    }
    // SAFETY: NULL is nothing.
    unsafe {
        assert!(rvg_frame_target_retain(std::ptr::null_mut()).is_null());
        rvg_frame_target_release(std::ptr::null_mut());
    }
}

/// A retain is one more reference to the same target: after the first of its two releases the
/// target is still usable, and its texture goes at the second, not before
#[test]
fn a_retained_target_lives_until_its_last_release() {
    let rig = Rig::new(1);
    let textures = || rig.p.world().live("MTLTexture");
    let before = textures();
    let (s, t) = target(rig.device, 4, 3, RVG_PIXEL_FORMAT_BGRA8_UNORM);
    assert_eq!(s, RVG_OK);
    let made: Vec<_> = textures()
        .into_iter()
        .filter(|id| !before.contains(id))
        .collect();
    assert_eq!(made.len(), 1, "{made:?}");
    let mut buf = [0xa5u8; 48];
    // SAFETY: the target made above, whose two references are given back once each; `buf` holds
    // 48 bytes.
    unsafe {
        assert_eq!(rvg_frame_target_retain(t), t);
        rvg_frame_target_release(t);
        assert!(textures().contains(&made[0]));
        assert_eq!(
            rvg_frame_target_read_pixels(t, buf.as_mut_ptr().cast(), 16, 48),
            RVG_OK
        );
        rvg_frame_target_release(t);
    }
    assert!(buf.iter().all(|&b| b == 0));
    assert!(!textures().contains(&made[0]));
}

/// The copy's extent is checked before anything is read
#[test]
fn read_pixels_checks_the_extent() {
    let rig = Rig::new(1);
    let (_, t) = target(rig.device, 4, 3, RVG_PIXEL_FORMAT_RGBA8_UNORM);
    // Rows of 16 bytes, 20 apart: the extent is 56.
    let mut buf = vec![0xa5u8; 64];
    let p = buf.as_mut_ptr().cast::<c_void>();
    // SAFETY: the target is live; `buf` holds 64 bytes.
    unsafe {
        assert_eq!(
            rvg_frame_target_read_pixels(t, p, 15, 64),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            rvg_frame_target_read_pixels(t, p, 20, 55),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            rvg_frame_target_read_pixels(t, std::ptr::null_mut(), 20, 64),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            rvg_frame_target_read_pixels(std::ptr::null(), p, 20, 64),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(rvg_frame_target_read_pixels(t, p, 20, 56), RVG_OK);
        rvg_frame_target_release(t);
    }
    // The fake platform's pixels read as zero, over the extent only.
    assert!(buf[..56].iter().all(|&b| b == 0));
    assert!(buf[56..].iter().all(|&b| b == 0xa5));
}

/// A frame is encoded uncommitted, committed once, and waited on after its commit; it keeps
/// the display alive until it completes, and the target keeps nothing of the device
#[test]
fn frame_states() {
    let mut rig = Rig::new(1);
    let d = display(rig.device);
    let (_, t) = target(rig.device, 8, 8, RVG_PIXEL_FORMAT_BGRA8_UNORM);
    let mut f = std::ptr::null_mut();
    let r = RvgRegion {
        x: 0,
        y: 0,
        width: 4,
        height: 4,
    };
    // SAFETY: live display and target; `f` is writable.
    unsafe {
        assert_eq!(
            rvg_display_encode_frame(d, std::ptr::null_mut(), &r, &mut f),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            rvg_display_encode_frame(d, t, &r, std::ptr::null_mut()),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(rvg_display_encode_frame(d, t, &r, &mut f), RVG_OK);
        assert_eq!(rvg_frame_wait(f), RVG_ERR_FRAME_STATE);
        assert_eq!(rvg_frame_commit(f), RVG_OK);
        assert_eq!(rvg_frame_commit(f), RVG_ERR_FRAME_STATE);
        // The fake GPU completes what was committed when told.
        assert_eq!(rig.p.world().complete_all(CompleteOn::Caller), 1);
        assert_eq!(rvg_frame_wait(f), RVG_OK);
        assert_eq!(
            rvg_frame_commit(std::ptr::null_mut()),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            rvg_frame_wait(std::ptr::null_mut()),
            RVG_ERR_INVALID_ARGUMENT
        );
        rvg_frame_release(f);
        rvg_frame_release(std::ptr::null_mut());
        assert_eq!(
            rvg_display_encode_frame(d, t, std::ptr::null(), &mut f),
            RVG_OK
        );
        rvg_display_release(d);
    }
    rig.release();
    // The target outlives the device, and the uncommitted frame the display.
    // SAFETY: the frame and target made above.
    unsafe {
        rvg_frame_release(f);
        rvg_frame_target_release(t);
    }
    assert_eq!(rig.ctx.rec.destroys().len(), 1);
}
