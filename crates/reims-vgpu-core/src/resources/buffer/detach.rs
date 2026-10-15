//! The detach buffer: a buffer made over a temporary VM allocation whose backing is detached
//! at once, then replaced by the guest's address ranges at the first CPU prepare. A discard
//! detaches it again.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::guest::GuestPtr;
use crate::platform::device::GpuDevice;
use crate::platform::resource::Buffer;
use crate::platform::values::{BufferDesc, OptionSet};
use crate::platform::{Handle, Platform};
use crate::raise::RaiseResult;
use crate::raise::{Kept, keep_forever};
use crate::resources::buffer::BufferResourceDescriptor;
use crate::resources::resource::AnyResource;
use crate::resources::resource::{
    RESOURCE_OPTION_READ_ONLY, Resource, ResourceClass, ResourceKind, STORAGE_MODE_SHARED, TaskRef,
    features_of, platform_for, read_descriptor,
};
use crate::resources::{DeathKind, RaiseKind, raise};
use crate::rt::Hold;

/// A buffer whose backing is detached until the guest's address ranges replace it.
pub struct DetachBufferResource<P: Platform> {
    task: TaskRef<P>,
    /// Never `None` after a successful creation.
    buffer: Option<P::Buffer>,
    /// The record's read-only bit, which only picks the creation's option.
    is_read_only: bool,
    length: u64,
    page: u32,
    is_detached: AtomicBool,
}

impl<P: Platform> DetachBufferResource<P> {
    /// A detached buffer of the record's length. The private bit is ignored; a read-only one
    /// needs argument buffers (logged, `None`). A guest pin in the placement slot is passed to
    /// Metal; a zero slot gets the made address. A raise detaching the new buffer leaks it.
    #[expect(
        clippy::new_ret_no_self,
        reason = "a kind is made as its shared `Resource`"
    )]
    pub fn new(
        task: TaskRef<P>,
        descriptor: &[u8],
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        let platform = platform_for(&task);
        let d: BufferResourceDescriptor = read_descriptor(descriptor);
        let is_read_only = d.is_read_only();
        if is_read_only && !features_of(&task).supports_argument_buffers {
            log::warn!("read-only buffer refused: the task's features have no argument buffers");
            return Ok(None);
        }
        // The descriptor and the VM allocation need the platform.
        let Some(p) = platform else {
            return Ok(None);
        };
        let mut options = vec![OptionSet::StorageMode(STORAGE_MODE_SHARED)];
        if is_read_only {
            options.push(OptionSet::ResourceOptionsOr(RESOURCE_OPTION_READ_ONLY));
        }
        // A guest pin (the first of two reads of the slot).
        let pin = placement
            .map(|pl| pl.read_u64(0, DeathKind::NullPlacement))
            .filter(|pin| *pin != 0);
        let allocation = match p.vm().allocate_anywhere(d.length) {
            Ok(a) => a,
            Err(error) => {
                log::error!(
                    "no address space for a detached buffer of {:#x} bytes: kern_return {}",
                    d.length,
                    error.0
                );
                return Ok(None);
            }
        };
        // The deallocator: Metal calls it when the buffer is destroyed.
        let vm_owner = Arc::downgrade(&p);
        let desc = BufferDesc {
            length: d.length,
            options,
            pinned_gpu_address: pin,
            contents: Some(contents(allocation, d.length)),
            no_copy: true,
            deallocator: Some(Box::new(move |pointer, length| {
                if let Some(p) = vm_owner.upgrade() {
                    let _ = p.vm().deallocate(pointer, length);
                }
            })),
        };
        // A raise leaves nothing made.
        let buffer = match task.upgrade() {
            Some(t) => t
                .mtl_device()
                .map_or(Ok(None), |device| device.new_buffer(desc))?,
            None => None,
        };
        // Written back only into a zero slot, and before the nil check (a failed creation
        // writes 0).
        if let Some(pl) = placement
            && pl.read_u64(0, DeathKind::NullPlacement) == 0
        {
            let address = buffer.as_ref().map_or(0, Buffer::gpu_address);
            pl.write_u64(0, address, DeathKind::NullPlacement);
        }
        // Nothing made, no log. Whether Metal ran the deallocator for a buffer it did not make
        // is unknown.
        let Some(buffer) = buffer else {
            return Ok(None);
        };
        // A raise keeps the buffer for good: its GPU VA is already in the guest's placement.
        if let Err(e) = buffer.detach_backing() {
            keep_forever(buffer, Kept::MetalNumbering);
            return Err(e);
        }
        Ok(Some(Resource::new(ResourceKind::DetachBuffer(
            DetachBufferResource {
                task,
                buffer: Some(buffer),
                is_read_only,
                length: d.length,
                page: d.page,
                is_detached: AtomicBool::new(true),
            },
        ))))
    }

    /// The task.
    pub fn task(&self) -> &TaskRef<P> {
        &self.task
    }

    /// The record's read-only bit.
    pub fn is_read_only(&self) -> bool {
        self.is_read_only
    }

    fn detached(&self) -> bool {
        self.is_detached.load(Ordering::Relaxed)
    }

    /// Replaces a detached backing with the guest's address ranges, writable as the page table
    /// says; Metal's refusal (or a torn-down task's missing ranges) raises and leaves it
    /// detached.
    fn ensure_physical(&self, hold: Hold<'_>) -> RaiseResult<()> {
        if !self.detached() {
            return Ok(());
        }
        let mut read_only = false;
        let ranges = match self.task.upgrade() {
            Some(t) => {
                t.address_ranges_for_virtual_page(hold, self.page, self.length, &mut read_only)?
            }
            None => None,
        };
        let replaced = match (&self.buffer, &ranges) {
            (Some(b), Some(r)) => b.replace_backing(r, read_only)?,
            // No buffer answers false.
            (None, _) => false,
            // Only a torn-down task gives no ranges (every task makes an array): taken as
            // Metal's refusal.
            (Some(_), None) => false,
        };
        if !replaced {
            return raise(RaiseKind::BufferBackingRefused);
        }
        self.is_detached.store(false, Ordering::Relaxed);
        Ok(())
    }

    fn buffer(&self) -> RaiseResult<P::Buffer> {
        match &self.buffer {
            Some(b) => Ok(b.clone_ref()),
            None => raise(RaiseKind::BufferInvalidated),
        }
    }
}

/// The no-copy contents of a detached buffer: `allocation`, `length` bytes.
#[allow(unsafe_code)] // architecture 17.3: the one unsafe constructor of host memory
fn contents<'a>(allocation: crate::host::HostPtr, length: u64) -> crate::platform::HostRange<'a> {
    // SAFETY: `allocation` is the fresh `mach_vm_allocate`d range of the buffer's length; nothing
    // else frees it, and the deallocator that goes with it is what releases it when Metal
    // destroys the buffer.
    unsafe { crate::platform::HostRange::new(allocation, length) }
}

impl<P: Platform> ResourceClass<P> for DetachBufferResource<P> {
    fn initial_guest_valid(&self) -> bool {
        true
    }

    fn get_resource(&self, this: &Resource<P>) -> RaiseResult<Option<AnyResource<P>>> {
        let _ = this;
        Ok(Some(AnyResource::Buffer(self.buffer()?)))
    }

    fn get_buffer(&self, this: &Resource<P>) -> RaiseResult<Option<P::Buffer>> {
        let _ = this;
        Ok(Some(self.buffer()?))
    }

    fn is_host_valid(&self, this: &Resource<P>) -> bool {
        let _ = this;
        // Derived from the backing; the host flag is dead.
        !self.detached()
    }

    fn prepare_with_cpu(&self, hold: Hold<'_>, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        self.ensure_physical(hold)
    }

    fn synchronize_with_cpu(&self, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        Ok(())
    }

    fn discard(&self, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        self.buffer
            .as_ref()
            .map_or(Ok(()), |b| b.detach_backing())?;
        self.is_detached.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn needs_sync(&self, this: &Resource<P>) -> bool {
        let _ = this;
        false
    }

    fn set_is_guest_valid(&self, this: &Resource<P>, v: bool) {
        let _ = (this, v);
    }

    fn is_guest_valid(&self, this: &Resource<P>) -> bool {
        let _ = this;
        true
    }

    fn needs_prepare(&self, this: &Resource<P>) -> bool {
        let _ = this;
        self.detached()
    }

    fn needs_cpu_paging(&self, this: &Resource<P>) -> bool {
        let _ = this;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::os::vm::KernReturn;
    use crate::platform::resource::GpuResource;
    use crate::resources::Op;
    use crate::resources::buffer::tests::desc;
    use crate::resources::resource::test_support::{Env, gone_task, kind, logged, placement};
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::catch_death;
    use crate::testkit::objects::{Fk, Model, kind};
    use crate::testkit::platform::FakePlatform;
    use crate::testkit::script::Step;

    type Fp = FakePlatform;

    fn detach(env: &Env, len: u64, flags: u8, pl: Option<GuestPtr>) -> Option<Arc<Resource<Fp>>> {
        env.run(|| DetachBufferResource::new(env.task(), &desc(len, 9, flags), pl))
            .unwrap()
    }

    fn parts(r: &Resource<Fp>) -> &DetachBufferResource<Fp> {
        match &r.kind {
            ResourceKind::DetachBuffer(k) => k,
            _ => panic!("kind"),
        }
    }

    /// The value the last buffer was made from, as the creator logged it.
    fn made_from(env: &Env) -> String {
        match env.p.call_log().calls_to("newBufferWithDescriptor:").last() {
            Some(c) => match &c.args[0] {
                Arg::Val(text) => text.clone(),
                x => panic!("{x:?}"),
            },
            None => panic!("no buffer made"),
        }
    }

    fn provide_ranges(env: &Env, page: u32) -> Fk<kind::AddressRangeArray> {
        let a = env
            .p
            .world()
            .new_obj::<kind::AddressRangeArray>(Model::None);
        env.rec.provide(u64::from(page), &a);
        a
    }

    mod init {
        use super::*;

        #[test]
        fn makes_the_buffer_over_the_allocation_and_detaches() {
            let env = Env::new();
            let r = detach(&env, 0x8000, 0, None).unwrap();
            let log = env.p.call_log();
            assert_eq!(
                log.sels(),
                [
                    "mach_vm_allocate",
                    "mtl_device",
                    "newBufferWithDescriptor:",
                    "detachBacking"
                ]
            );
            // Shared, no pin, no copy of the allocation, and its deallocator.
            let made = made_from(&env);
            assert!(
                made.starts_with(
                    "BufferDesc { length: 32768, options: [StorageMode(0)], \
                     pinned_gpu_address: None, contents: Some("
                ),
                "{made}"
            );
            assert!(
                made.ends_with("no_copy: true, deallocator: true }"),
                "{made}"
            );
            let b = r.get_buffer().unwrap().unwrap();
            assert_eq!((b.length(), b.resource_options()), (0x8000, 0));
            assert!(r.state.guest_valid() && !r.state.host_valid());
            assert!(parts(&r).detached() && !r.is_host_valid() && r.needs_prepare());
        }

        #[test]
        fn read_only_needs_argument_buffers() {
            let env = Env::new();
            assert!(detach(&env, 0x4000, 2, None).is_none());
            assert_eq!(env.p.call_log().count("mach_vm_allocate"), 0);
            env.set_features(|f| f.supports_argument_buffers = true);
            let r = detach(&env, 0x4000, 2, None).unwrap();
            assert!(parts(&r).is_read_only());
            // The read-only bit or'ed into the shared options.
            assert!(
                made_from(&env).contains("options: [StorageMode(0), ResourceOptionsOr(16777216)]")
            );
            let b = r.get_buffer().unwrap().unwrap();
            assert_eq!(b.resource_options(), 0x100_0000);
        }

        #[test]
        fn read_only_on_a_gone_task_dies_reading_its_features() {
            let d =
                catch_death(|| DetachBufferResource::<Fp>::new(gone_task(), &desc(16, 1, 2), None));
            assert_eq!(d.err(), Some(DeathKind::GoneTaskFeatures.site()));
        }

        #[test]
        fn a_guest_pin_is_passed_and_not_written_back() {
            // The made address goes only into a zero slot, before the nil check.
            let env = Env::new();
            let (m, pl) = placement(0x1_0000_4000);
            let r = detach(&env, 0x4000, 0, Some(pl)).unwrap();
            assert!(made_from(&env).contains("pinned_gpu_address: Some(4294983680)"));
            assert_eq!(m.read_u64(0), Some(0x1_0000_4000));
            let b = r.get_buffer().unwrap().unwrap();
            assert_eq!(b.gpu_address(), 0x1_0000_4000);
        }

        #[test]
        fn write_back_only_into_a_zero_slot() {
            let env = Env::new();
            let (m, pl) = placement(0);
            let r = detach(&env, 0x4000, 0, Some(pl)).unwrap();
            let b = r.get_buffer().unwrap().unwrap();
            assert_ne!(b.gpu_address(), 0);
            assert_eq!(m.read_u64(0), Some(b.gpu_address()));
            // A failed creation writes nil's 0.
            let (m, pl) = placement(0);
            env.p.script().push("newBufferWithDescriptor:", Step::Nil);
            assert!(detach(&env, 0x4000, 0, Some(pl)).is_none());
            assert_eq!(m.read_u64(0), Some(0));
        }

        #[test]
        fn a_failed_allocation_logs_and_makes_nothing() {
            let env = Env::new();
            env.p
                .script()
                .push("mach_vm_allocate", Step::value(KernReturn(3)));
            let (made, lines) = logged(|| detach(&env, 0x4000, 0, None));
            assert!(made.is_none());
            assert_eq!(
                lines,
                vec!["no address space for a detached buffer of 0x4000 bytes: kern_return 3"]
            );
            assert_eq!(env.p.call_log().count("newBufferWithDescriptor:"), 0);
        }

        #[test]
        fn the_deallocator_frees_the_allocation_when_the_buffer_dies() {
            let env = Env::new();
            let r = detach(&env, 0x4000, 0, None).unwrap();
            let addr = r.get_buffer().unwrap().unwrap().contents().addr();
            assert_eq!(env.p.call_log().count("mach_vm_deallocate"), 0);
            drop(r);
            let c = env.p.call_log().calls_to("mach_vm_deallocate");
            assert_eq!(c.len(), 1);
            assert_eq!(c[0].args, vec![Arg::Ptr(addr), Arg::U64(0x4000)]);
        }

        #[test]
        fn without_a_platform_scope_nothing_is_made() {
            let env = Env::new();
            assert!(
                DetachBufferResource::new(env.task(), &desc(16, 1, 0), None)
                    .unwrap()
                    .is_none()
            );
        }

        #[test]
        fn a_detach_raise_leaks_the_buffer() {
            let env = Env::new();
            env.p.script().push(
                "detachBacking",
                Step::raise("NSInvalidArgumentException", "x"),
            );
            assert!(
                env.run(|| DetachBufferResource::new(env.task(), &desc(0x4000, 1, 0), None))
                    .is_err()
            );
            // The leaked buffer never runs its deallocator.
            assert_eq!(env.p.call_log().count("mach_vm_deallocate"), 0);
        }
    }

    mod paging {
        use super::*;

        #[test]
        fn prepare_replaces_the_backing_once_with_the_page_tables_writability() {
            let env = Env::new();
            let r = detach(&env, 0x4000, 0, None).unwrap();
            let ranges = provide_ranges(&env, 9);
            r.prepare_with_cpu(Hold::NONE).unwrap();
            let a = env.rec.log().calls_to("address_ranges_for_virtual_page");
            assert_eq!(
                a[0].args,
                vec![Arg::U32(9), Arg::U64(0x4000), Arg::Bool(false)]
            );
            let c = env
                .p
                .call_log()
                .calls_to("replaceBackingWithRanges:readOnly:");
            assert_eq!(c[0].args, vec![Arg::obj(&ranges), Arg::Bool(false)]);
            assert!(r.is_host_valid() && !r.needs_prepare());
            r.prepare_with_cpu(Hold::NONE).unwrap();
            assert_eq!(env.rec.log().count("address_ranges_for_virtual_page"), 1);
            let m = env.p.call_log().mark();
            r.synchronize_with_cpu().unwrap();
            assert!(env.p.call_log().since(m).is_empty());
        }

        #[test]
        fn a_refused_replacement_raises_and_stays_detached() {
            let env = Env::new();
            let r = detach(&env, 0x4000, 0, None).unwrap();
            provide_ranges(&env, 9);
            env.p
                .script()
                .push("replaceBackingWithRanges:readOnly:", Step::value(false));
            let hold = Hold::NONE;
            assert_eq!(
                kind(r.prepare_with_cpu(hold)),
                Some(RaiseKind::BufferBackingRefused)
            );
            assert!(r.needs_prepare());
        }

        #[test]
        fn no_ranges_is_the_same_refusal() {
            let env = Env::new();
            let r = detach(&env, 0x4000, 0, None).unwrap();
            let hold = Hold::NONE;
            assert_eq!(
                kind(r.prepare_with_cpu(hold)),
                Some(RaiseKind::BufferBackingRefused)
            );
            assert_eq!(
                env.p.call_log().count("replaceBackingWithRanges:readOnly:"),
                0
            );
        }

        #[test]
        fn a_range_raise_propagates() {
            let env = Env::new();
            let r = detach(&env, 0x4000, 0, None).unwrap();
            env.p.script().push(
                "address_ranges_for_virtual_page",
                Step::raise("NSGenericException", "Address out of range for task"),
            );
            let hold = Hold::NONE;
            assert!(r.prepare_with_cpu(hold).is_err());
            assert_eq!(
                env.p.call_log().count("replaceBackingWithRanges:readOnly:"),
                0
            );
        }

        #[test]
        fn discard_detaches_again_and_the_next_prepare_rebinds() {
            let env = Env::new();
            let r = detach(&env, 0x4000, 0, None).unwrap();
            provide_ranges(&env, 9);
            r.prepare_with_cpu(Hold::NONE).unwrap();
            r.discard().unwrap();
            assert!(!r.is_host_valid() && r.needs_prepare());
            assert_eq!(env.p.call_log().count("detachBacking"), 2);
            r.prepare_with_cpu(Hold::NONE).unwrap();
            assert_eq!(env.rec.log().count("address_ranges_for_virtual_page"), 2);
        }
    }

    mod accessors {
        use super::*;

        #[test]
        fn validity_is_derived_and_the_guest_setter_is_a_no_op() {
            // Host validity is derived from the backing.
            let env = Env::new();
            let r = detach(&env, 0x4000, 0, None).unwrap();
            r.set_is_guest_valid(false);
            assert!(r.is_guest_valid() && r.state.guest_valid());
            r.set_is_host_valid(true);
            assert!(!r.is_host_valid(), "the flag is dead");
            assert!(!r.needs_sync() && r.needs_cpu_paging() && r.supports_discard());
            let b = r.get_buffer().unwrap().unwrap();
            assert!(
                matches!(r.get_resource().unwrap(), Some(AnyResource::Buffer(x)) if x.id() == b.id())
            );
            let hold = Hold::NONE;
            assert_eq!(
                kind(r.prepare_in_encoder(hold, None)),
                Some(RaiseKind::Unrecognized(Op::Prepare))
            );
            assert_eq!(
                kind(r.get_heap()),
                Some(RaiseKind::Unrecognized(Op::GetHeap))
            );
        }

        #[test]
        fn nil_buffer_getters_raise() {
            let r = Resource::new(ResourceKind::DetachBuffer(DetachBufferResource::<Fp> {
                task: gone_task(),
                buffer: None,
                is_read_only: false,
                length: 0,
                page: 0,
                is_detached: AtomicBool::new(false),
            }));
            assert_eq!(kind(r.get_resource()), Some(RaiseKind::BufferInvalidated));
            assert_eq!(kind(r.get_buffer()), Some(RaiseKind::BufferInvalidated));
            // A nil buffer's discard sends to nil and still marks detached.
            r.discard().unwrap();
            assert!(r.needs_prepare());
        }
    }
}
