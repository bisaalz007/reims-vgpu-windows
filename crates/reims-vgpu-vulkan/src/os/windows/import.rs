//! Writable twins of read-only section views. The runtime drops them after freeing the import.

use super::vm::{self, WindowsVm};
use crate::os::backing as files;
use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm as _, KernReturn};
use reims_vgpu_core::sync::Mutex;
use std::collections::BTreeMap;
use std::os::windows::io::OwnedHandle;
use std::ptr::NonNull;
use std::sync::{Arc, Weak};

static ZEROS: Mutex<Vec<Weak<OwnedHandle>>> = Mutex::new(Vec::new());
static TWINS: Mutex<BTreeMap<u64, (u64, u64)>> = Mutex::new(BTreeMap::new());

pub(crate) const TWINS_READ_ONLY: bool = true;

pub(crate) fn allocate_zeros(length: u64) -> Result<HostAddr, KernReturn> {
    let at = WindowsVm.allocate_anywhere(length)?;
    if let Some(piece) = files::pieces(at.addr().to_u64(), 1).and_then(|p| p.into_iter().next()) {
        let mut zeros = ZEROS.lock();
        zeros.retain(|file| file.strong_count() != 0);
        zeros.push(Arc::downgrade(&piece.backing.file));
    }
    Ok(at)
}

fn is_zero(file: &Arc<OwnedHandle>) -> bool {
    ZEROS
        .lock()
        .iter()
        .any(|weak| weak.upgrade().is_some_and(|zero| Arc::ptr_eq(&zero, file)))
}

#[derive(Debug)]
struct Twin {
    first: u64,
    length: u64,
}
impl Drop for Twin {
    fn drop(&mut self) {
        TWINS.lock().remove(&self.first);
        let _ = WindowsVm.deallocate(HostAddr::from_addr(self.first.to_usize()), self.length);
    }
}

#[derive(Debug)]
pub(crate) struct ImportView {
    at: NonNull<u8>,
    twin: Option<Twin>,
}
impl ImportView {
    pub(crate) fn at(&self) -> NonNull<u8> {
        self.at
    }
    pub(crate) fn writes_kept(&self) -> bool {
        self.twin.is_some()
    }
    pub(crate) fn release(
        self,
        original: NonNull<u8>,
        inner: Option<rt::Deallocator>,
    ) -> (Option<rt::Deallocator>, Option<rt::KeepAlive>) {
        let Some(twin) = self.twin else {
            return (inner, None);
        };
        let original = SendPtr(original);
        let inner = inner.map(|inner| -> rt::Deallocator {
            Box::new(move |_, length| inner(original.get(), length))
        });
        (inner, Some(Box::new(twin)))
    }
}

struct SendPtr(NonNull<u8>);
// SAFETY: bookkeeping only, returned to the original deallocator under its contract.
unsafe impl Send for SendPtr {}
impl SendPtr {
    fn get(self) -> NonNull<u8> {
        self.0
    }
}

pub(crate) fn import_view(at: NonNull<u8>, length: u64) -> ImportView {
    let plain = ImportView { at, twin: None };
    let start = at.as_ptr().addr().to_u64();
    let page = WindowsVm.page_size();
    let first = start & !(page - 1);
    let Some(end) = start
        .checked_add(length.max(1))
        .and_then(|end| end.checked_next_multiple_of(page))
    else {
        return plain;
    };
    let Some(mut parts) = files::pieces(first, end - first) else {
        return plain;
    };
    if parts.iter().all(|part| part.prot & 2 != 0) {
        return plain;
    }
    let length = end - first;
    let Ok(base) = WindowsVm.allocate_anywhere(length) else {
        return plain;
    };
    let twin = Twin {
        first: base.addr().to_u64(),
        length,
    };
    let Some(own) = files::pieces(twin.first, length).and_then(|p| p.into_iter().next()) else {
        return plain;
    };
    for part in &mut parts {
        if is_zero(&part.backing.file) {
            part.backing = files::Backing {
                file: Arc::clone(&own.backing.file),
                offset: part.at,
            };
        }
        part.prot = 3;
    }
    if vm::map_parts(twin.first, length, &parts).is_err() {
        return plain;
    }
    TWINS.lock().insert(twin.first, (length, first));
    let pointer = std::ptr::with_exposed_provenance_mut((twin.first + start - first).to_usize());
    let Some(at) = NonNull::new(pointer) else {
        return plain;
    };
    ImportView {
        at,
        twin: Some(twin),
    }
}

pub(crate) fn shown_at(at: NonNull<u8>) -> NonNull<u8> {
    let address = at.as_ptr().addr().to_u64();
    let twins = TWINS.lock();
    let Some((&twin, &(length, original))) = twins.range(..=address).next_back() else {
        return at;
    };
    if address >= twin + length {
        return at;
    }
    NonNull::new(std::ptr::with_exposed_provenance_mut(
        (original + address - twin).to_usize(),
    ))
    .unwrap_or(at)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::platform::os::vm::VmProt;
    #[test]
    fn a_read_only_view_has_a_writable_alias_kept_until_the_import_release() {
        let source = WindowsVm.allocate_anywhere(4096).unwrap();
        unsafe { source.as_ptr::<u8>().write(5) };
        WindowsVm.protect_max(source, 4096, VmProt::READ).unwrap();
        let pointer = NonNull::new(source.as_ptr::<u8>()).unwrap();
        let view = import_view(pointer, 4096);
        assert!(view.writes_kept());
        assert_ne!(view.at(), pointer);
        assert_eq!(shown_at(view.at()), pointer);
        unsafe {
            view.at().as_ptr().write(8);
            assert_eq!(pointer.as_ptr().read(), 8);
        }
        let alias = HostAddr::from_ptr(view.at().as_ptr());
        let (_, keep) = view.release(pointer, None);
        assert!(WindowsVm.deallocate(source, 4096).is_success());
        unsafe {
            assert_eq!(alias.as_ptr::<u8>().read(), 8);
        }
        drop(keep);
        assert!(!WindowsVm.readable(alias, 1));
    }
    #[test]
    fn a_zeros_twin_has_its_own_zeros_and_cannot_change_the_original() {
        let source = allocate_zeros(4096).unwrap();
        WindowsVm.protect_max(source, 4096, VmProt::READ).unwrap();
        let pointer = NonNull::new(source.as_ptr::<u8>()).unwrap();
        let view = import_view(pointer, 4096);
        assert!(view.writes_kept());
        unsafe {
            view.at().as_ptr().write(9);
            assert_eq!(pointer.as_ptr().read(), 0);
        }
        drop(view);
        assert!(WindowsVm.deallocate(source, 4096).is_success());
    }
}
