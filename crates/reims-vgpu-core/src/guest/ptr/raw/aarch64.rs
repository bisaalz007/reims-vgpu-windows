//! The raw accesses. Every function here is given an address `check` has passed.

use std::sync::atomic::AtomicU32;

use crate::num::Truncate;

use crate::host::HostPtr;

/// One volatile byte load.
pub(super) fn load_u8(a: HostPtr) -> u8 {
    // SAFETY: `a` is outside the zero region (`check`). That it is mapped is the caller's
    // precondition: guest RAM stays mapped while its mapping lives.
    unsafe { a.as_ptr::<u8>().read_volatile() }
}

/// One volatile byte store.
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

/// One unaligned 16-bit `ldrh`: one access, single-copy atomic under FEAT_LSE2 when it stays
/// inside a 16-byte granule, never split into bytes.
pub(super) fn load_u16_unaligned(a: HostPtr) -> u16 {
    let v: u32;
    // SAFETY: as `load_u8`; arm64 Normal memory allows an unaligned `ldrh` (arm64 only).
    unsafe {
        core::arch::asm!("ldrh {v:w}, [{a}]", a = in(reg) a.as_ptr::<u8>(), v = out(reg) v,
            options(nostack, preserves_flags, readonly));
    }
    v.low_u16()
}

/// One unaligned 32-bit `ldr` (see `load_u16_unaligned`).
pub(super) fn load_u32_unaligned(a: HostPtr) -> u32 {
    let v: u32;
    // SAFETY: as `load_u16_unaligned`.
    unsafe {
        core::arch::asm!("ldr {v:w}, [{a}]", a = in(reg) a.as_ptr::<u8>(), v = out(reg) v,
            options(nostack, preserves_flags, readonly));
    }
    v
}

/// One unaligned 64-bit `ldr` (see `load_u16_unaligned`).
pub(super) fn load_u64_unaligned(a: HostPtr) -> u64 {
    let v: u64;
    // SAFETY: as `load_u16_unaligned`.
    unsafe {
        core::arch::asm!("ldr {v:x}, [{a}]", a = in(reg) a.as_ptr::<u8>(), v = out(reg) v,
            options(nostack, preserves_flags, readonly));
    }
    v
}

/// One unaligned 16-bit `strh` (see `load_u16_unaligned`).
pub(super) fn store_u16_unaligned(a: HostPtr, v: u16) {
    // SAFETY: as `load_u16_unaligned`.
    unsafe {
        core::arch::asm!("strh {v:w}, [{a}]", a = in(reg) a.as_ptr::<u8>(),
            v = in(reg) u32::from(v), options(nostack, preserves_flags));
    }
}

/// One unaligned 32-bit `str` (see `load_u16_unaligned`).
pub(super) fn store_u32_unaligned(a: HostPtr, v: u32) {
    // SAFETY: as `load_u16_unaligned`.
    unsafe {
        core::arch::asm!("str {v:w}, [{a}]", a = in(reg) a.as_ptr::<u8>(), v = in(reg) v,
            options(nostack, preserves_flags));
    }
}

/// One unaligned 64-bit `str` (see `load_u16_unaligned`).
pub(super) fn store_u64_unaligned(a: HostPtr, v: u64) {
    // SAFETY: as `load_u16_unaligned`.
    unsafe {
        core::arch::asm!("str {v:x}, [{a}]", a = in(reg) a.as_ptr::<u8>(), v = in(reg) v,
            options(nostack, preserves_flags));
    }
}

/// One `stp wLo, wHi, [a]`: clang's store of an 8-byte `{uint32_t, uint32_t}` pair as one
/// instruction. Under FEAT_LSE2 it is one single-copy-atomic access when its eight bytes
/// sit inside one 16-byte granule, whatever their alignment; across a granule it is one
/// access per register. The same instruction at every alignment, so no caller reasons
/// about which case it is in.
pub(super) fn stp_u32_pair(a: HostPtr, lo: u32, hi: u32) {
    // SAFETY: as `load_u16_unaligned`; arm64 Normal memory allows an unaligned `stp`.
    unsafe {
        core::arch::asm!("stp {lo:w}, {hi:w}, [{a}]", a = in(reg) a.as_ptr::<u8>(),
            lo = in(reg) lo, hi = in(reg) hi, options(nostack, preserves_flags));
    }
}

/// One `ldp wLo, wHi, [a]`: clang's load of a `{uint32_t, uint32_t}` pair as one
/// instruction (see `stp_u32_pair` for its atomicity).
pub(super) fn ldp_u32_pair(a: HostPtr) -> (u32, u32) {
    let lo: u32;
    let hi: u32;
    // SAFETY: as `stp_u32_pair`.
    unsafe {
        core::arch::asm!("ldp {lo:w}, {hi:w}, [{a}]", a = in(reg) a.as_ptr::<u8>(),
            lo = out(reg) lo, hi = out(reg) hi, options(nostack, preserves_flags, readonly));
    }
    (lo, hi)
}

/// One `stp xzr, xzr, [a]`: clang's `memset(p, 0, 16)` (and a 16-byte `= {}`). One store,
/// single-copy atomic as a whole at a 16-byte-aligned address under FEAT_LSE2, and the same
/// instruction at any alignment.
pub(super) fn stp_zero_16(a: HostPtr) {
    // SAFETY: as `load_u16_unaligned`; arm64 Normal memory allows an unaligned `stp`.
    unsafe {
        core::arch::asm!("stp xzr, xzr, [{a}]", a = in(reg) a.as_ptr::<u8>(),
            options(nostack, preserves_flags));
    }
}

/// A misaligned `ldapr` (acquire, RCpc), what Rust's `Acquire` load emits; the caller checked
/// that it stays inside one 16-byte granule, where FEAT_LSE2 performs it.
pub(super) fn ldapr_u32_unaligned(a: HostPtr) -> u32 {
    let v: u32;
    // SAFETY: as `load_u16_unaligned`; inside one granule (`within_granule`), so no fault.
    unsafe {
        core::arch::asm!("ldapr {v:w}, [{a}]", a = in(reg) a.as_ptr::<u8>(), v = out(reg) v,
            options(nostack, preserves_flags, readonly));
    }
    v
}

/// A misaligned `ldar` (seq_cst, RCsc), clang's `std::atomic<uint32_t>::load()`.
pub(super) fn ldar_u32_unaligned(a: HostPtr) -> u32 {
    let v: u32;
    // SAFETY: as `ldapr_u32_unaligned`.
    unsafe {
        core::arch::asm!("ldar {v:w}, [{a}]", a = in(reg) a.as_ptr::<u8>(), v = out(reg) v,
            options(nostack, preserves_flags, readonly));
    }
    v
}

/// A misaligned `stlr`, clang's release and seq_cst `store`.
pub(super) fn stlr_u32_unaligned(a: HostPtr, v: u32) {
    // SAFETY: as `ldapr_u32_unaligned`.
    unsafe {
        core::arch::asm!("stlr {v:w}, [{a}]", a = in(reg) a.as_ptr::<u8>(), v = in(reg) v,
            options(nostack, preserves_flags));
    }
}

/// A misaligned `ldsetal`, clang's seq_cst `fetch_or` (`|=`); returns the old value.
pub(super) fn ldsetal_u32_unaligned(a: HostPtr, v: u32) -> u32 {
    let old: u32;
    // SAFETY: as `ldapr_u32_unaligned` (FEAT_LSE2 covers the LSE atomics inside a granule).
    unsafe {
        core::arch::asm!("ldsetal {v:w}, {old:w}, [{a}]", a = in(reg) a.as_ptr::<u8>(),
            v = in(reg) v, old = out(reg) old, options(nostack, preserves_flags));
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
