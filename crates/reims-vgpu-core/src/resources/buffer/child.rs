//! The heap buffer: a buffer a heap made in itself. It holds only the Metal buffer (which holds
//! the heap), is never paged, and answers a discard with nothing.

use std::sync::Arc;

use crate::platform::{Handle, Platform};
use crate::raise::RaiseResult;
use crate::resources::resource::AnyResource;
use crate::resources::resource::{Resource, ResourceClass, ResourceKind};
use crate::rt::Hold;

/// A buffer made in a heap; discardable when the heap is a detach heap.
pub struct HeapBufferResource<P: Platform> {
    buffer: P::Buffer,
    discardable: bool,
}

impl<P: Platform> HeapBufferResource<P> {
    /// A child of `heap` over `buffer` (the caller's reference moves in). Neither validity flag
    /// is set.
    ///
    /// The child keeps the heap's Metal object (through the buffer), not the heap resource, so
    /// deleting the heap frees its transfer buffers and temporary allocation while the child keeps
    /// working.
    #[expect(
        clippy::new_ret_no_self,
        reason = "a kind is made as its shared `Resource`"
    )]
    pub fn new(heap: Arc<Resource<P>>, buffer: P::Buffer, discardable: bool) -> Arc<Resource<P>> {
        // The heap resource is not kept (measured, testing.md 7.1).
        drop(heap);
        Resource::new(ResourceKind::HeapBuffer(HeapBufferResource {
            buffer,
            discardable,
        }))
    }

    /// Whether a discard may be asked of it.
    pub fn discardable(&self) -> bool {
        self.discardable
    }
}

impl<P: Platform> ResourceClass<P> for HeapBufferResource<P> {
    fn prepare_in_encoder(
        &self,
        _hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        let _ = (this, encoder);
        Ok(())
    }

    fn synchronize_in_encoder(
        &self,
        _hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        let _ = (this, encoder);
        Ok(())
    }

    fn discard(&self, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        Ok(())
    }

    fn get_resource(&self, this: &Resource<P>) -> RaiseResult<Option<AnyResource<P>>> {
        let _ = this;
        Ok(Some(AnyResource::Buffer(self.buffer.clone_ref())))
    }

    fn get_buffer(&self, this: &Resource<P>) -> RaiseResult<Option<P::Buffer>> {
        let _ = this;
        Ok(Some(self.buffer.clone_ref()))
    }

    fn needs_sync(&self, this: &Resource<P>) -> bool {
        let _ = this;
        false
    }

    fn needs_prepare(&self, this: &Resource<P>) -> bool {
        let _ = this;
        false
    }

    fn supports_discard(&self, this: &Resource<P>) -> bool {
        let _ = this;
        self.discardable
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::device::GpuDevice;
    use crate::raise::OnFault;
    use crate::resources::resource::test_support::{Env, kind};
    use crate::resources::{Op, RaiseKind};
    use crate::testkit::platform::FakePlatform;

    type Fp = FakePlatform;

    fn child(env: &Env, discardable: bool) -> (Arc<Resource<Fp>>, <Fp as Platform>::Buffer) {
        let b = env
            .p
            .device()
            .new_buffer_of_length(OnFault::Die, 64, 0)
            .unwrap()
            .unwrap();
        let heap = env
            .run(|| {
                crate::resources::buffer::private::PrivateBufferResource::new(
                    env.task(),
                    &[64, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    None,
                )
            })
            .unwrap()
            .unwrap();
        let r = HeapBufferResource::new(heap, b.clone_ref(), discardable);
        (r, b)
    }

    /// The child holds the buffer, not the heap resource
    #[test]
    fn holds_the_buffer_not_the_heap_resource() {
        // A child holds its base's Metal object, not the base resource.
        let env = Env::new();
        let (r, b) = child(&env, false);
        assert!(!r.state.guest_valid() && !r.state.host_valid());
        assert_eq!(b.retain_count(), 2);
        // The heap resource went at `new`: only the child and `b` hold `b`.
        drop(r);
        assert_eq!(b.retain_count(), 1);
    }

    /// Never paged; a discard does nothing
    #[test]
    fn never_paged_and_discard_is_a_no_op() {
        let env = Env::new();
        let (r, b) = child(&env, false);
        let m = env.p.call_log().mark();
        r.prepare_in_encoder(Hold::NONE, None).unwrap();
        r.synchronize_in_encoder(Hold::NONE, None).unwrap();
        r.discard().unwrap();
        assert!(env.p.call_log().since(m).is_empty());
        assert!(!r.needs_sync() && !r.needs_prepare());
        assert_eq!(r.get_buffer().unwrap().map(|x| x.id()), Some(b.id()));
        assert!(
            matches!(r.get_resource().unwrap(), Some(AnyResource::Buffer(x)) if x.id() == b.id())
        );
        // The flags are the default's: set by opcode 52, read back.
        r.set_is_host_valid(true);
        assert!(r.is_host_valid());
    }

    /// Discardability is the heap's
    #[test]
    fn discardability_follows_the_heap() {
        let env = Env::new();
        assert!(!child(&env, false).0.supports_discard());
        assert!(child(&env, true).0.supports_discard());
        assert_eq!(child(&env, true).0.kind_name(), "discardable heap buffer");
    }

    /// No heap and no CPU paging
    #[test]
    fn heap_and_cpu_operations_are_not_answered() {
        let env = Env::new();
        let (r, _b) = child(&env, true);
        let hold = Hold::NONE;
        assert_eq!(
            kind(r.get_heap()),
            Some(RaiseKind::Unrecognized(Op::GetHeap))
        );
        assert_eq!(
            kind(r.prepare_with_cpu(hold)),
            Some(RaiseKind::Unrecognized(Op::PrepareCpu))
        );
        assert!(!r.needs_cpu_paging());
    }
}
