//! The raw accesses on x86-64. Every function here is given an address `check` has passed.
//!
//! The functions keep the names `GuestPtr` calls them by, which are arm64's instructions
//! (`stp_u32_pair`, `ldar_u32_unaligned`, …). Each is one x86-64 instruction of the same width
//! that gives the guest what that arm64 instruction gives it:
//!
//! - An access of up to 8 bytes is one `mov`, aligned or not. Both vendors perform one inside an
//!   aligned 8-byte quadword atomically (Intel's SDM vol. 3A, "Guaranteed Atomic Operations";
//!   AMD's APM vol. 2, "Access Atomicity"), and Intel one anywhere inside a 64-byte cache line,
//!   so on Intel one inside a 16-byte granule is single-copy atomic as arm64's is under
//!   FEAT_LSE2. On AMD that is assumed for one that crosses an 8-byte boundary inside the
//!   granule, which its manual does not promise.
//! - A 16-byte clear at a 16-byte-aligned address is one `movaps`, which both vendors perform
//!   atomically on cacheable memory on a part with AVX (the same sections), as `stp` is under
//!   FEAT_LSE2 there.
//! - x86-64 orders every load as an acquire and every store as a release, so a misaligned
//!   acquire or seq_cst load is a plain `mov`. The misaligned store, which serves release and
//!   seq_cst stores alike, is an `xchg`, a full fence, so a later seq_cst load is never performed
//!   ahead of it, as `ldar` never passes `stlr`. The one read-modify-write is a `lock cmpxchg`
//!   loop. `GuestPtr` lets a misaligned atomic act only inside a 16-byte granule, so no locked
//!   access crosses a cache line. An aligned atomic is Rust's of the ordering `GuestPtr` names.

use std::sync::atomic::AtomicU32;

use crate::num::Truncate;

use crate::host::HostPtr;

/// One volatile byte load: one `mov`.
pub(super) fn load_u8(a: HostPtr) -> u8 {
    // SAFETY: `a` is outside the zero region (`check`). That it is mapped is the caller's
    // precondition: guest RAM stays mapped while its mapping lives.
    unsafe { a.as_ptr::<u8>().read_volatile() }
}

/// One volatile byte store: one `mov`.
pub(super) fn store_u8(a: HostPtr, v: u8) {
    // SAFETY: as `load_u8`.
    unsafe { a.as_ptr::<u8>().write_volatile(v) }
}

/// One volatile 16-bit load of an aligned address.
pub(super) fn load_u16(a: HostPtr) -> u16 {
    // SAFETY: as `load_u8`; the caller checked the alignment.
    unsafe { a.as_ptr::<u16>().read_volatile() }
}

/// One volatile 16-bit store to an aligned address.
pub(super) fn store_u16(a: HostPtr, v: u16) {
    // SAFETY: as `load_u16`.
    unsafe { a.as_ptr::<u16>().write_volatile(v) }
}

/// One volatile 32-bit load of an aligned address.
pub(super) fn load_u32(a: HostPtr) -> u32 {
    // SAFETY: as `load_u16`.
    unsafe { a.as_ptr::<u32>().read_volatile() }
}

/// One volatile 32-bit store to an aligned address.
pub(super) fn store_u32(a: HostPtr, v: u32) {
    // SAFETY: as `load_u16`.
    unsafe { a.as_ptr::<u32>().write_volatile(v) }
}

/// One volatile 64-bit load of an aligned address.
pub(super) fn load_u64(a: HostPtr) -> u64 {
    // SAFETY: as `load_u16`.
    unsafe { a.as_ptr::<u64>().read_volatile() }
}

/// One volatile 64-bit store to an aligned address.
pub(super) fn store_u64(a: HostPtr, v: u64) {
    // SAFETY: as `load_u16`.
    unsafe { a.as_ptr::<u64>().write_volatile(v) }
}

/// One unaligned 16-bit `movzx`: one access, never split into bytes, atomic where the module docs
/// say.
pub(super) fn load_u16_unaligned(a: HostPtr) -> u16 {
    let v: u32;
    // SAFETY: as `load_u8`; x86-64 allows an unaligned `movzx`.
    unsafe {
        core::arch::asm!("movzx {v:e}, word ptr [{a}]", a = in(reg) a.as_ptr::<u8>(),
            v = out(reg) v, options(nostack, preserves_flags, readonly));
    }
    v.low_u16()
}

/// One unaligned 32-bit `mov` (see `load_u16_unaligned`).
pub(super) fn load_u32_unaligned(a: HostPtr) -> u32 {
    let v: u32;
    // SAFETY: as `load_u16_unaligned`; x86-64 allows an unaligned `mov`.
    unsafe {
        core::arch::asm!("mov {v:e}, dword ptr [{a}]", a = in(reg) a.as_ptr::<u8>(),
            v = out(reg) v, options(nostack, preserves_flags, readonly));
    }
    v
}

/// One unaligned 64-bit `mov` (see `load_u16_unaligned`).
pub(super) fn load_u64_unaligned(a: HostPtr) -> u64 {
    let v: u64;
    // SAFETY: as `load_u32_unaligned`.
    unsafe {
        core::arch::asm!("mov {v}, qword ptr [{a}]", a = in(reg) a.as_ptr::<u8>(),
            v = out(reg) v, options(nostack, preserves_flags, readonly));
    }
    v
}

/// One unaligned 16-bit `mov` (see `load_u16_unaligned`).
pub(super) fn store_u16_unaligned(a: HostPtr, v: u16) {
    // SAFETY: as `load_u32_unaligned`.
    unsafe {
        core::arch::asm!("mov word ptr [{a}], {v:x}", a = in(reg) a.as_ptr::<u8>(),
            v = in(reg) v, options(nostack, preserves_flags));
    }
}

/// One unaligned 32-bit `mov` (see `load_u16_unaligned`).
pub(super) fn store_u32_unaligned(a: HostPtr, v: u32) {
    // SAFETY: as `load_u32_unaligned`.
    unsafe {
        core::arch::asm!("mov dword ptr [{a}], {v:e}", a = in(reg) a.as_ptr::<u8>(),
            v = in(reg) v, options(nostack, preserves_flags));
    }
}

/// One unaligned 64-bit `mov` (see `load_u16_unaligned`).
pub(super) fn store_u64_unaligned(a: HostPtr, v: u64) {
    // SAFETY: as `load_u32_unaligned`.
    unsafe {
        core::arch::asm!("mov qword ptr [{a}], {v}", a = in(reg) a.as_ptr::<u8>(),
            v = in(reg) v, options(nostack, preserves_flags));
    }
}

/// A `{u32, u32}` pair stored as one 64-bit `mov`, `lo` at `a`: one access at every alignment,
/// as arm64's `stp` is, atomic as a whole where the module docs say (assumed on AMD across an
/// 8-byte boundary), so a guest's 64-bit load of the pair inside one 16-byte granule sees the old
/// pair or the new one.
pub(super) fn stp_u32_pair(a: HostPtr, lo: u32, hi: u32) {
    let v = u64::from(hi).wrapping_shl(32) | u64::from(lo);
    // SAFETY: as `load_u32_unaligned`.
    unsafe {
        core::arch::asm!("mov qword ptr [{a}], {v}", a = in(reg) a.as_ptr::<u8>(),
            v = in(reg) v, options(nostack, preserves_flags));
    }
}

/// A `{u32, u32}` pair loaded as one 64-bit `mov` (see `stp_u32_pair` for its atomicity):
/// `(word at +0, word at +4)`.
pub(super) fn ldp_u32_pair(a: HostPtr) -> (u32, u32) {
    let v: u64;
    // SAFETY: as `load_u32_unaligned`.
    unsafe {
        core::arch::asm!("mov {v}, qword ptr [{a}]", a = in(reg) a.as_ptr::<u8>(),
            v = out(reg) v, options(nostack, preserves_flags, readonly));
    }
    (v.low_u32(), v.wrapping_shr(32).low_u32())
}

/// Sixteen zero bytes as one store, as arm64's `stp xzr, xzr` is: `movaps` at a 16-byte-aligned
/// address, single-copy atomic there on a part with AVX, and `movups` at any other, one
/// instruction with no such promise, as `stp` has none there either.
pub(super) fn stp_zero_16(a: HostPtr) {
    if a.addr() & 0xf == 0 {
        // SAFETY: as `load_u32_unaligned`; `movaps` is given a 16-byte-aligned address.
        unsafe {
            core::arch::asm!("xorps {z}, {z}", "movaps xmmword ptr [{a}], {z}",
                a = in(reg) a.as_ptr::<u8>(), z = out(xmm_reg) _,
                options(nostack, preserves_flags));
        }
    } else {
        // SAFETY: as `load_u32_unaligned`; `movups` takes any alignment.
        unsafe {
            core::arch::asm!("xorps {z}, {z}", "movups xmmword ptr [{a}], {z}",
                a = in(reg) a.as_ptr::<u8>(), z = out(xmm_reg) _,
                options(nostack, preserves_flags));
        }
    }
}

/// A misaligned acquire load: a plain `mov`, which x86-64 orders before every later load and
/// store. The caller checked that it stays inside one 16-byte granule, as on arm64. Unlike the
/// plain loads it is not `readonly`, so the compiler moves no later access ahead of it either.
pub(super) fn ldapr_u32_unaligned(a: HostPtr) -> u32 {
    let v: u32;
    // SAFETY: as `load_u32_unaligned`.
    unsafe {
        core::arch::asm!("mov {v:e}, dword ptr [{a}]", a = in(reg) a.as_ptr::<u8>(),
            v = out(reg) v, options(nostack, preserves_flags));
    }
    v
}

/// A misaligned seq_cst load: a plain `mov`, as clang's `std::atomic<uint32_t>::load()` is on
/// x86-64, where the seq_cst and release stores (`stlr_u32_unaligned`'s `xchg`) carry the fence.
pub(super) fn ldar_u32_unaligned(a: HostPtr) -> u32 {
    ldapr_u32_unaligned(a)
}

/// A misaligned release or seq_cst store: one `xchg`, clang's seq_cst `store` on x86-64. Inside
/// one granule (the caller's check) it never crosses a cache line, so it is never a split lock.
pub(super) fn stlr_u32_unaligned(a: HostPtr, v: u32) {
    // SAFETY: as `load_u32_unaligned`; inside one granule, so inside one cache line.
    unsafe {
        core::arch::asm!("xchg dword ptr [{a}], {v:e}", a = in(reg) a.as_ptr::<u8>(),
            v = inout(reg) v => _, options(nostack, preserves_flags));
    }
}

/// A misaligned seq_cst `fetch_or` (`|=`): clang's `lock cmpxchg` loop on x86-64, one atomic
/// read-modify-write of the word, retried while the guest changes it in between. Returns the old
/// value.
pub(super) fn ldsetal_u32_unaligned(a: HostPtr, v: u32) -> u32 {
    let old: u32;
    // SAFETY: as `stlr_u32_unaligned`.
    unsafe {
        core::arch::asm!(
            "mov eax, dword ptr [{a}]",
            "2:",
            "mov {t:e}, eax",
            "or {t:e}, {v:e}",
            "lock cmpxchg dword ptr [{a}], {t:e}",
            "jne 2b",
            a = in(reg) a.as_ptr::<u8>(), v = in(reg) v, t = out(reg) _, out("eax") old,
            options(nostack),
        );
    }
    old
}

/// The `std::atomic<uint32_t>` at an aligned address.
pub(super) fn atomic_u32<'a>(a: HostPtr) -> &'a AtomicU32 {
    // SAFETY: as `load_u16`; the caller checked 4-byte alignment. The reference is to an
    // atomic cell, which the guest writing concurrently does not invalidate, and it is used
    // for the one access and dropped: no reference outlives the access.
    unsafe { AtomicU32::from_ptr(a.as_ptr::<u32>()) }
}
