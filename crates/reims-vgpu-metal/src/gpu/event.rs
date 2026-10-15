//! Events: core's `SharedEvent`, and the event, handle and listener handle types.

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLEvent, MTLSharedEvent, MTLSharedEventHandle, MTLSharedEventListener};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::raise::RaiseResult;

use crate::msg::{self, Policy};
use crate::platform::{MetalPlatform, Obj};
use crate::spi::methods::SharedEventSpi;

/// `id<MTLEvent>`.
pub type MetalEvent = Obj<ProtocolObject<dyn MTLEvent>>;

/// `id<MTLSharedEvent>`.
pub type MetalSharedEvent = Obj<ProtocolObject<dyn MTLSharedEvent>>;

impl reims_vgpu_core::platform::event::SharedEvent<MetalPlatform> for MetalSharedEvent {
    /// `newSharedEventHandle`: the `+1` handle, or nil.
    fn new_shared_event_handle(&self) -> Option<<MetalPlatform as Platform>::SharedEventHandle> {
        msg::get("newSharedEventHandle", || {
            self.newSharedEventHandle_or_nil()
        })
        .map(Obj::from_retained)
    }
    fn set_signaled_value(&self, signaled_value: u64) -> RaiseResult<()> {
        msg::send("setSignaledValue:", Policy::Caught, || {
            self.setSignaledValue(signaled_value)
        })
    }
    /// `waitUntilSignaledValue:timeoutMS:`, which blocks up to `milliseconds`.
    fn wait_until_signaled_value(&self, value: u64, milliseconds: u64) -> bool {
        msg::get("waitUntilSignaledValue:timeoutMS:", || {
            self.waitUntilSignaledValue_timeoutMS(value, milliseconds)
        })
    }
}

/// `MTLSharedEventHandle *`.
pub type MetalSharedEventHandle = Obj<MTLSharedEventHandle>;

/// `MTLSharedEventListener *`.
pub type MetalSharedEventListener = Obj<MTLSharedEventListener>;

#[cfg(test)]
mod tests {

    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::platform::event::SharedEvent;

    use super::*;
    use crate::gpu::resource::testkit::*;

    /// setSignaledValue: sets the value; waitUntilSignaledValue:timeoutMS: is YES for a reached
    /// value and NO after the timeout for an unreached one
    #[test]
    fn signal_and_wait() {
        let d = dev();
        let e = d.new_shared_event().unwrap();
        e.set_signaled_value(3).unwrap();
        assert_eq!(e.get().signaledValue(), 3);
        assert!(e.wait_until_signaled_value(3, 0));
        assert!(e.wait_until_signaled_value(2, 0));
        assert!(!e.wait_until_signaled_value(4, 10));
    }

    /// newSharedEventHandle gives a +1 handle that opens the same event
    #[test]
    fn new_shared_event_handle() {
        let d = dev();
        let e = d.new_shared_event().unwrap();
        e.set_signaled_value(9).unwrap();
        let h = e.new_shared_event_handle().unwrap();
        let e2 = d.open_shared_event(Some(&h)).unwrap();
        assert_eq!(e2.get().signaledValue(), 9);
    }
}
