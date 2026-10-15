//! The Objective-C runtime's autorelease pool, which core's `rt::pool::Pool` pushes and pops
//! around each unit of work (D22): what Metal autoreleases inside one is freed at its end. Core's
//! own values never go to it: those a release point holds are in core's holders (`rt::held`).
//!
//! What a `+0` result leaves in the pool depends on the build. objc2 retains a `+0` result for
//! the handle, and in an optimized build (the product's, the suite's) the runtime's return-value
//! handshake elides Metal's own autorelease of it: the handle holds the only reference. So an
//! uncommitted command buffer from `-commandBuffer` goes at its handle's drop, with its slot among
//! the queue's 64, and buffers dropped uncommitted inside one pool never block the queue
//! (`dropped_uncommitted_command_buffers_in_one_pool_do_not_block`). A dev build cannot elide it:
//! Metal's autoreleased reference stays in the thread's innermost pool, and each dropped buffer
//! keeps its slot until that pool pops (`a_pool_per_unit_frees_uncommitted_command_buffers`).

use std::ffi::c_void;

use reims_vgpu_core::platform::os::autorelease::{NativePool, install_native};

/// `objc_autoreleasePoolPush()`.
fn push() -> *mut c_void {
    // SAFETY: no preconditions; the token is popped on this thread by `pop`.
    unsafe { objc2::ffi::objc_autoreleasePoolPush() }
}

/// `objc_autoreleasePoolPop(token)`.
fn pop(token: *mut c_void) {
    // SAFETY: `token` came from `push` on this thread and has not been popped (core pops a
    // token once, and never one an outer pop already took).
    unsafe { objc2::ffi::objc_autoreleasePoolPop(token) }
}

/// Installs the real pool for core's pools (once per process).
pub fn install() {
    install_native(NativePool { push, pop });
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::raise::OnFault;
    use reims_vgpu_core::rt::Pool;

    /// A pool around each unit of work frees what Metal autoreleased inside it at its end, in a
    /// dev build too (info_errors, blit_err_wrong_refs, mutgap_dec_render_fault_releases_encoder,
    /// interplay_decodewalk_11/12: the 65th uncommitted command buffer blocked forever)
    #[test]
    fn a_pool_per_unit_frees_uncommitted_command_buffers() {
        use reims_vgpu_core::platform::command::CommandQueue;
        use reims_vgpu_core::platform::device::GpuDevice;
        install();
        let Some(dev) = crate::MetalPlatform::system_default_device() else {
            return;
        };
        let (tx, rx) = std::sync::mpsc::channel();
        #[allow(clippy::disallowed_methods)] // a host thread: no pool of ours, as the test needs
        std::thread::spawn(move || {
            let queue = dev.new_command_queue().unwrap();
            for _ in 0..70 {
                let _pool = Pool::here();
                let cb = queue.command_buffer(OnFault::Die).unwrap().unwrap();
                drop(cb); // never committed
            }
            let _ = tx.send(());
        });
        assert!(
            rx.recv_timeout(std::time::Duration::from_secs(20)).is_ok(),
            "the 65th -commandBuffer blocked: uncommitted buffers were never freed"
        );
    }

    /// The one-pool form. `-commandBuffer` answers its buffer autoreleased, and the handle adds
    /// no reference of its own to the pool. An optimized build (the product's, the suite's)
    /// elides Metal's autorelease, so the handle holds the buffer's only reference: 70 buffers
    /// made and dropped uncommitted inside one pool do not block, since each buffer, and its slot
    /// among the queue's 64, goes at its drop. A dev build cannot elide it: the pool keeps each
    /// buffer to its pop (`a_pool_per_unit_frees_uncommitted_command_buffers`), and only the
    /// count is checked
    #[test]
    fn dropped_uncommitted_command_buffers_in_one_pool_do_not_block() {
        use reims_vgpu_core::platform::command::CommandQueue;
        use reims_vgpu_core::platform::device::GpuDevice;
        fn retain_count(o: &objc2::runtime::AnyObject) -> usize {
            // SAFETY: `-retainCount` of a live object.
            unsafe { objc2::msg_send![o, retainCount] }
        }
        install();
        let Some(dev) = crate::MetalPlatform::system_default_device() else {
            return;
        };
        let (tx, rx) = std::sync::mpsc::channel();
        #[allow(clippy::disallowed_methods)] // a host thread: no pool of ours, as the test needs
        std::thread::spawn(move || {
            let queue = dev.new_command_queue().unwrap();
            objc2::rc::autoreleasepool(|_| {
                let first = queue.command_buffer(OnFault::Die).unwrap().unwrap();
                // SAFETY: a command buffer is an Objective-C object.
                let count = retain_count(unsafe {
                    &*objc2::rc::Retained::as_ptr(first.as_retained()).cast()
                });
                drop(first); // never committed
                let _ = tx.send(count);
                if count == 1 {
                    for _ in 1..70 {
                        let cb = queue.command_buffer(OnFault::Die).unwrap().unwrap();
                        drop(cb); // never committed
                    }
                    let _ = tx.send(0);
                }
            });
        });
        let wait = std::time::Duration::from_secs(20);
        match rx.recv_timeout(wait) {
            // The handle's reference alone: Metal's autorelease was elided.
            Ok(1) => assert!(
                rx.recv_timeout(wait).is_ok(),
                "the 65th -commandBuffer blocked: the pool kept the dropped buffers"
            ),
            // The handle's and Metal's autoreleased one.
            Ok(count) => assert_eq!(count, 2),
            Err(e) => panic!("no command buffer: {e}"),
        }
    }
}
