# Windows native audit

Source and GitHub state inspected on 2026-10-12. This is a read-only audit of
the revisions below, for a native Windows host and an x86-64 macOS Tahoe
guest. It does not report a Windows build, a GPU test or a guest boot.
The concurrent Windows implementation is outside this document's baseline.

## Findings that affect the current port

The current tree needs a Windows implementation of its OS boundary and its
guest-memory backing. Reapplying the old Windows feature gates is not enough.
Reims PR #57 was merged, but commit
[`98fe2a3bd5db3b3d87a501c2b1153486980a0569`](https://github.com/steelbrain/reims-vgpu/commit/98fe2a3bd5db3b3d87a501c2b1153486980a0569)
replaced the tree with `libreims_vgpu` and its C API on 2026-10-11.
The old fork's Rust modules, QEMU shim and versioned shim ABI are different
from the ones the current host calls.

Read from the audited source:

| Boundary | Current implementation and exact gap |
|---|---|
| Public platform | `crates/reims-vgpu/src/lib.rs` defines `DefaultPlatform` only for macOS and Linux. Its Cargo feature is now `vulkan`, not the old fork's `backend-vulkan,host-window`. |
| OS services and RAM | `crates/reims-vgpu-vulkan/src/os.rs` selects only `os/macos.rs` and `os/linux.rs`. It imports Unix `OwnedFd` unconditionally. `GuestRamFile` and the public `MemoryFile` in `crates/reims-vgpu/src/device.rs` hold `Arc<OwnedFd>`. The Linux implementation builds aliases from shared file mappings in `os/linux/{files,vm,import}.rs`. There is no Windows counterpart. |
| C memory-file validation | `crates/reims-vgpu-c/src/device.rs` uses Unix descriptors, `fstat`, `sysconf` and `fcntl(F_DUPFD_CLOEXEC)`. `RvgMemoryRangeFile` is a fixed-layout API record; an OS handle cannot silently replace its `fd` field. [The API's additive versioning rules](api.md) apply to any new backing representation. |
| Callback boundary | `crates/reims-vgpu-c/src/ffi/catch.rs` selects only macOS and Linux. Linux's implementation uses `catch_unwind`; the Windows choice is missing. The library panic hook and callback thread/order contract must remain intact. |
| Window and presentation | `crates/reims-vgpu-window/src/native.rs` and `crates/reims-vgpu/src/window/vulkan.rs` select only macOS and Linux. Linux's SDL3/Vulkan implementation exists, but Windows input, window lifetime and WSI wiring do not. `reims-vgpu-vulkan/src/os/wsi.rs` asks `host_os` for the loader and surface extensions; that Windows provider is missing too. |
| Distribution | `xtask/src/dist.rs` selects only `dist/{macos,linux}.rs` and uses Unix `PermissionsExt`. Linux dist writes `.a`/`.so` artifacts. Windows archive/DLL naming and native static-link requirements need a Windows distribution path. |
| QEMU configuration | At the QEMU pin, `meson.build:882` discovers `reims-vgpu` through pkg-config only when `host_os` is `darwin` or `linux`. Windows is excluded before the new library can be linked. |
| QEMU RAM backing | `hw/display/apple-gfx.c`, `apple_gfx_memory_range_file`, recognizes a shared file descriptor and the host's page size. `apple_gfx_memory_files` rejects missing backing on Linux, and passes NULL elsewhere. Neither function implements a Windows section-backed guest-memory contract. Merely adding Windows to Meson does not supply that backing. |
| Runtime host import | At the metal2vulkan pin, `crates/metal2vulkan-runtime/src/host_memory.rs:248` returns `None` from `host_page()` on non-Unix hosts. `Importer::new` at line 468 then refuses host memory as not implemented. Reims relies on these no-copy buffers, so fixing only Reims's module gates does not close this runtime gap. |
| Runtime exit | `crates/metal2vulkan-runtime/src/exit_guard.rs` installs its exit hook only on Unix; the non-Unix `install()` is empty. This is source evidence of absent Windows exit handling, not a measured Windows exit crash. |

These are implementation gaps read from code, not compiler diagnostics
produced by this audit. All current Reims paths above are at
[`9b2c5452`](https://github.com/steelbrain/reims-vgpu/tree/9b2c5452de94a474cba28d80d7f99d9d7b19ffe3),
QEMU paths at
[`1a72df98`](https://github.com/steelbrain/qemu-reims-vgpu/tree/1a72df984914fb9929d35b1217619178133d65a4),
and runtime paths at
[`13b279df`](https://github.com/steelbrain/metal2vulkan/tree/13b279df1bf87893d389f955a90c55b6ff3d0f9b).

## Revisions compared

Refs were resolved to commits before reading them. Fork sources were read
selectively with `git show`/`git grep` from existing repositories; no fork
source checkout or submodule initialization was made for this audit.

| Repository/ref | Full commit | Relationship to the audit |
|---|---|---|
| steelbrain/reims-vgpu master | `9b2c5452de94a474cba28d80d7f99d9d7b19ffe3` | Requested baseline; audit worktree started here. |
| MAXZVER/reims-vgpu windows | `73098237b3bfdd8e7006fcc4b2a8e48d67cd8eb7` | Local `maxzver/windows`; merge base with baseline is `69a57dd69a6958e946c03b73e02db331f330f435`. |
| tqmane/reims-vgpu master | `9a5299a0b77202c5d9e7262efe216faca1f5c8b2` | Older source layout; its Vulkan OS allowlist is macOS/Linux. |
| tqmane/reims-vgpu steelbrain/stability-fixes-v2 | `65fd08e968b3f5f59a26bbcdfc979094f1b3eac5` | Older Windows implementation; merge base with baseline is `8178caa7b9a6ec4c394453f9e4f7e28d3529f27c`. |
| steelbrain/metal2vulkan Reims dependency pin | `13b279df1bf87893d389f955a90c55b6ff3d0f9b` | Both translator and runtime in the current workspace use this pin. The existing local clone was at this commit when inspected. |
| steelbrain/qemu-reims-vgpu Reims submodule pin | `1a72df984914fb9929d35b1217619178133d65a4` | Current `apple-gfx` on `reims_vgpu.h`; existing local clone was at this commit when inspected. |
| MAXZVER/qemu-reims-vgpu windows | `e92847870dea4760355889edac3d623af05a9223` | Live GitHub branch resolution and matching local Git object; older `reims-vgpu-pci` shim. |
| tqmane/qemu-reims-vgpu bot-branch-host-reims-vgpu-vmapple | `61f0506060996e68632730d153c4e3543910eece` | QEMU commit actually pinned by the tqmane stability fork. |
| tqmane/qemu-reims-vgpu host-reims-vgpu-vmapple | `999c6d791995a4f6cff79a2473b596bca64bfacc` | Older alternative branch; not the stability fork's pin. |
| tqmane/qemu-reims-vgpu master | `300438ffbb8d9430cac2fcc15cba6f482b2c0587` | Older QEMU master; not the stability fork's pin. |

The MAXZVER Reims branch still records `bd88218da09b86ed9c78bf5f9354168812a7ba6b`
as its QEMU submodule, even though its README describes combining it with the
separate Windows QEMU branch. That pairing cannot be inferred from
`vendor/qemu` alone.

### What the older forks actually supply

At `73098237`, MAXZVER retains the old `backend/vulkan/engine` implementation.
Its `guest_ram_map.rs` provides `write_refusal` and
`surface_write_refusal`; `mapping_write/vulkan.rs` calls the latter for
`licence_mapper_ref_texture_surface`. `reims-vgpu-config/src/lib.rs`
defines `REIMS_VGPU_GUEST_IMPORT_WRITES` and the narrower
`REIMS_VGPU_GUEST_IMPORT_SURFACE_WRITES`. Its engine contains page-diff
writeback in `backend/vulkan/engine/page_diff.rs`, backed by its own shader.
These are implemented old-engine mechanisms, not switches in the current
runtime-backed platform.

The same fork still contains `DependencyGraph` in
`crates/reims-vgpu-core/src/depend.rs` and `ResidentTargets` in
`backend/vulkan/engine/resource_lease.rs`. The lease acquisition removes
only an exact identity before inserting the new one. It still has the test
`a_resource_retains_each_child_identity_until_the_resource_ends`, so the
generation replacement proposed in #111 is not present there.
Its README expressly separates lab results from validation of the exact
branch, says the lab carried the #78 graph-compaction patch that the branch
does not include, and says Tahoe had not been tested in that lab. The later
Tahoe report in #70 used `windows-tahoe`, a different ref. Those results
are not validation of `windows` at `73098237`.
[Source: pinned MAXZVER README](https://github.com/MAXZVER/reims-vgpu/blob/73098237b3bfdd8e7006fcc4b2a8e48d67cd8eb7/README.md).

At tqmane's `65fd08e9`, the old `src/lib.rs` explicitly admits Windows for
`backend-vulkan`. `backend/vulkan/engine/context.rs` has a Windows WSI arm;
`host_window/present.rs` uses winit's Windows `with_any_thread` opt-in.
Its own metal2vulkan translator dependency is
`tqmane/metal2vulkan@ea26f79f84c98174dbafdc794fdd543fee784097`, not the current
runtime pin. The tqmane master uses
`steelbrain/metal2vulkan@f9832cec3fe55d28069e3c8aa4bd307a7c0296c9`; MAXZVER
windows uses `steelbrain/metal2vulkan@9e0e99a41dc3cb8bb7e288b531f1698a79fd4b1c`.
Reusing any of their translation results does not validate `13b279df`.

At tqmane QEMU `61f05060`, `hw/display/meson.build` has a Windows static-link
arm and the old `reims_vgpu_qemu_abi.h` include path.
`hw/display/reims-vgpu-pci.c` maps only host-contiguous page runs and declares
them stable; scattered pages are refused, and unmap does nothing for these
borrowed RAMBlock pointers. That is a different backing strategy from the
current Linux VM's file-backed alias construction.

MAXZVER QEMU `e9284787` contains the XSAVE-validity/SYSENTER work, timer
resolution opt-out, asynchronous harvest with `harvests_settled`, and WHPX
dirty tracking discussed below. Its PCI shim includes
`reims_vgpu_qemu_abi.h` and checks its ABI version at realize.
It cannot be linked to the new `rvg_*` library as an interchangeable host.
The current fixed/sized struct rules are in [docs/api.md](api.md); adding
Windows does not authorize changing existing offsets or removing statuses.

## Requested Reims issues and PR

GitHub states below come from REST issue/pull-request responses retrieved
on 2026-10-12. A closed issue or an open issue is not, by itself, evidence
of current implementation coverage. Historical failure reports are
attributed to their reported builds.

| Item and API state | Source coverage at the audited baseline | Missing native Windows evidence |
|---|---|---|
| [#17 Support Windows](https://github.com/steelbrain/reims-vgpu/issues/17): **closed**, completed, 2026-08-31 | The maintainer's closing comment follows the initial Windows merge. The subsequent `98fe2a3b` replacement has the Windows module gaps listed above. Closure does not establish support in `9b2c5452`. | Native library/distribution build, C linkage, window and WHPX/Tahoe boot on the new architecture. |
| [PR #57 Windows host port](https://github.com/steelbrain/reims-vgpu/pull/57): **merged**, 2026-08-31 | Merge commit `517aae658d4cfe8ef90450e005066020661d316c` is an ancestor of `9b2c5452` (`merge-base --is-ancestor` succeeds). Changed old `backend/vulkan/engine/context.rs`, `host_window/present.rs`, `src/lib.rs`, `scripts/qemu-build/qemu-build.sh`, `vm/boot-windows.sh`. These paths were replaced; the current split crates do not retain equivalent Windows selectors. No reason to reapply the old patch verbatim. | Compile/link and live Windows behavior of the new OS/WSI modules; PR-era first-frame evidence is not that test. |
| [#70 Tahoe/import page-table panic](https://github.com/steelbrain/reims-vgpu/issues/70): **open** | The report concerns the old host-import engine. Current `gpu/device.rs::no_copy_release`, `remap.rs::VulkanVm` and `UnmapHold` explicitly track remaps and delay owned unmaps until imports are freed. The old `GUEST_IMPORT` switches are absent. This is lifetime machinery present in source, not a demonstrated fix of this guest panic. #70's later Windows report used a different MAXZVER ref and narrowed IOSurface writes. | Cold Tahoe boot plus active GPU load on the new Windows backing path, including teardown/reuse; capture a panic/freeze rather than treating missing serial output as success. |
| [#71 Windows/WHPX release panic](https://github.com/steelbrain/reims-vgpu/issues/71): **open** | Separate WHPX register/MMIO corruption and GPU-write lifetime hypotheses are discussed in the issue. QEMU's current XSAVE and MMIO paths still lack the protections confirmed below for PR #4. The current Reims path instead uses `memory/mapping.rs`, `gpu/device.rs` and `remap.rs`. The historical copying/import switches are unavailable here. | Windows WHPX stop/continue and normal execution, followed by task/surface release under load; independently establish host state integrity and guest-page lifetime. |
| [#87 dependency-graph tombstones](https://github.com/steelbrain/reims-vgpu/issues/87): **open** | `DependencyGraph` and its `depend.rs` site are absent from current `crates/`. The MAXZVER fork retains the old graph. Adding its old compaction fix to the new core would port a mechanism the new tree does not have. This does not prove every new admission path has bounded cost. | Sustained native guest work with latency/cost observations on the new FIFO implementation; historical graph benchmarks do not apply. |
| [#90 Tahoe white/tiled corruption](https://github.com/steelbrain/reims-vgpu/issues/90): **open** | The report at `69a57dd6` localized corruption before window presentation. Current display shaders are `reims-vgpu-metal/src/gpu/shaders.metal`, imported by `reims-vgpu-vulkan/src/gpu/shaders.rs`; GPU encoding/backing lives in the new Vulkan crate and runtime. [The accuracy ledger](accuracy-ledger.md) records remaining measured differences. There is no Windows/Tahoe comparison in this audit establishing this symptom is gone. | Compare guest capture, C-API frame readback and native window pixels after drawing, menus and window motion; a frame count alone cannot establish correct pixels. |
| [#93 Tahoe RenderBox/archive aborts](https://github.com/steelbrain/reims-vgpu/issues/93): **open** | Current [QEMU documentation](qemu.md#x86-64-macos-guests-metals-error-mode) records the x86 PCI guest plugin's nil `targetDeviceArchitecture` and a measured guest setting, `METAL_ERROR_MODE` integer **5**, in system GlobalPreferences. This is newer source-recorded evidence than the issue's integer-3 comment. The host does not select that guest plugin; GPU-family/AIR-answer changes are not a demonstrated repair. | On the intended Tahoe clone, verify iconservicesagent, Spotlight and dock.extra after login and reboot with the documented setting. Current recorded evidence is Linux/Tahoe 26.6.1, not this Windows/Tahoe build. |
| [#100 GPU writes into released pages](https://github.com/steelbrain/reims-vgpu/issues/100): **open** | Old `licence_gva_plane` and release-marker code are absent. Current `memory/mapping.rs` owns mappings by counted references; `gpu/device.rs::no_copy_release` gives the runtime keep-alives/holds; `remap.rs` reports address reuse. Existing tests include `an_unmap_of_held_pages_waits_for_the_hold` and `an_alias_is_the_guests_pages_both_ways`. These checks cover parts of mapping/import lifetime, not proof that a guest cannot recycle a page before a GPU store on Windows. | GPU stores versus guest unmap/reuse, source-alias release and same-address remap on the Windows implementation, including IOSurface and task deletion. The original report used an arm64 guest; it does not directly validate x86 Tahoe. |
| [#101 NoStamp/lazy writeback black regions](https://github.com/steelbrain/reims-vgpu/issues/101): **open** | `NoStamp`, `mapper_ref_texture_seed_guest_wrote` and `REIMS_VGPU_LAZY_WRITEBACK` are absent from current `crates/`. The specific old resident/writeback decision cannot be patched here. Both older forks still have a `NoStamp` arm; its presence alone does not prove the newer surrounding routes reproduce the issue. | Verify load/store continuity and CPU/GPU shared-memory updates after menus/dialogs close on Windows; do not cite removing a symbol as pixel validation. |
| [#111 resident generation pin growth](https://github.com/steelbrain/reims-vgpu/issues/111): **open** | Current `crates/` contains neither `ResidentTargets` nor `resident_target_backing_with`. MAXZVER `resource_lease.rs` still retains distinct generations beside each other; the proposed supersede rule is absent there. Current lifetime code is in the Vulkan backing/remap modules and metal2vulkan. Porting a fix to the removed registry is unnecessary. | A native long-lived desktop/workload and post-app-close reclamation observations. The issue's old Sequoia idle failure took hours; a short smoke run cannot disprove a comparable resource-growth problem. |

The symbol-absence checks above used `git grep` against the entire current
`crates/` tree, not only a guessed replacement file. Removing a defect's
implementation site is recorded as a retired path, never as a successful
reproduction/confirmation run.

## Requested QEMU PRs

All five requested PRs are **open and unmerged** in their API responses.
Their head SHAs are recorded below. GitHub's `merge_commit_sha` on an open
PR can be a test merge and is not treated as a merged commit.
For #4, all pages of the changed-file API were fetched: its large base diff
includes unrelated QEMU changes, so the first 100 files are insufficient.

| PR, inspected head | Actual paths and current coverage | Missing native evidence |
|---|---|---|
| [#4 WHPX Windows/AMD fixes](https://github.com/steelbrain/qemu-reims-vgpu/pull/4), `d06dd6e8087d3095c72f9c952849add7bd88b228` | Relevant files include `target/i386/whpx/{whpx-all,whpx-apic}.c`. At current `1a72df98`, `whpx_get_xsave_state` returns on failed/empty reads before freeing its temporary; register writeback invokes `whpx_set_xsave_state` without a per-VP validity flag. `whpx_handle_mmio` calls the emulator without a BQL guard. The APIC MSI path drops vector 0 without an INIT-delivery exception. These protections are present in MAXZVER's Windows fork and absent at the audited pin. The bundle also changes SYSENTER, CPUID/cache topology, TSC-deadline, MISC_ENABLE and page-fault delivery; this audit does not claim every tweak is required on every host CPU. | Native WHPX register/pause-resume integrity, exception/INIT behavior and cold guest boot on the actual host CPU. PR evidence is its author's AMD/Ventura setup. |
| [#6 hidden-window timer resolution](https://github.com/steelbrain/qemu-reims-vgpu/pull/6), `e11632c027666ded75e99c173c379b8a9dc8ce1e` | `os-win32.c` adds process power-throttling opt-outs after `timeBeginPeriod`. Current QEMU retains `timeBeginPeriod` without those opt-outs; MAXZVER windows has them. The old shim's heartbeat cited by the PR is gone; current library clocks/threads would be supplied by the new Windows OS module. This is an absent OS policy change, not measured evidence of a current Reims timer defect. | Visible, covered and minimized window timings under the host's normal policy; determine whether the new clock/display path actually has the PR's symptom. |
| [#7 asynchronous dirty harvest](https://github.com/steelbrain/qemu-reims-vgpu/pull/7), `669fafeabb52430160d7a9d02bec267f94ff9589` | Changes `hw/display/reims-vgpu-pci.c` and requires the old ABI v21 `ReimsVgpuHostOps.harvests_settled` from Reims #110. MAXZVER's fork contains that protocol. Current QEMU uses `apple-gfx.c` on `rvg_*`; there is no old shim or dirty-harvest protocol to move off the vCPU. Do not add an asynchronous wait or reorder the new core solely to replay this old-engine optimization. | If any equivalent work is introduced, prove its guest-memory ordering and teardown against the new C API. No existing PR-era timing validates that implementation. |
| [#10 WHPX dirty tracking/ranged harvest](https://github.com/steelbrain/qemu-reims-vgpu/pull/10), `e5e0194845c689a925dd7317935b005d7668f4a6` | `accel/whpx/whpx-common.c`, `include/system/{memory,whpx-internal}.h`, `system/{memory,physmem}.c`, and the removed `hw/display/reims-vgpu-dirty.c`. Current `whpx_log_sync` still marks the whole region dirty; it has no `WHvQueryGpaRangeDirtyBitmap` tracking path. `physical_memory_set_dirty_lebitmap` still uses `HOST_LONG_BITS` for its length/slow-path word stride instead of `BITS_PER_LONG`; MAXZVER windows carries the tracking and LLP64 fix. The general WHPX/bitmap fixes are separable from the obsolete Reims harvest caller. | Native bitmap correctness at unaligned offsets and Windows word sizes if this tracking path is used; preserve every dirty-log consumer. The new Reims host does not currently use the old tracked-surface API. |
| [#11 lockless BAR0 reads](https://github.com/steelbrain/qemu-reims-vgpu/pull/11), `c07a31ad969ed6b127aca93c71220436f0332e41` | Old `hw/display/reims-vgpu-pci.c`. The current equivalent already exists: `apple-gfx.c::apple_gfx_common_init` calls `memory_region_enable_lockless_io`; `apple_gfx_read` calls `rvg_device_mmio_read` directly. Commit `a771139e82` introduced the current path. The EFI BAR hook takes the BQL separately in `apple-gfx-pci.c::apple_gfx_pci_mmio_written`. Reapplying #11's old wrapper is unnecessary. | Windows WHPX dispatch behavior, concurrent accesses/reset and the new EFI hook. Existing source coverage does not establish WHPX's emulator honors the desired lockless fast path. |

## Evidence still needed from the native implementation

The missing checks are tied to the actual boundaries above, not a request
to run the old fork's entire suite. None was run by this documentation task.

| Boundary | Evidence required before claiming it works |
|---|---|
| Build/package/ABI | Native build with the chosen Windows Rust/C toolchain, actual static-link dependencies and dist paths; link a host against the generated current header and realize `apple-gfx-pci` paused. Header/layout checks must preserve old sized-config acceptance and fixed record layouts. |
| Guest backing | Shared aliases of scattered guest pages; OS allocation granularity versus guest pages; ownership after source alias release; handle/file retention; remap at an unchanged host address; writable/read-only imports. Existing Linux tests cannot run a missing Windows VM module. |
| GPU lifetime | Submitted writes while a task or surface ends, delayed import free, and released/reused guest pages. Test the Windows runtime import implementation separately from core mapping bookkeeping. |
| Window | Native Vulkan loader/ICD and Win32 surface, SDL event/input loop, mode/resize/DPI, cursor, close/teardown, and presentation correctness. Linux/macOS extension probes do not test Win32 WSI. |
| WHPX | Guest boot and pause/continue with live vector state, interrupts/INIT and MMIO; dirty bitmap tests only for a path that actually uses it. |
| Tahoe | Cold installer/recovery and installed-system boots on writable clones/overlays, serial and host fatal logs, guest/presented pixels and UI processes, then active-use and reclamation observations. Record the exact guest build: the existing repo's 26.6.1/26.6.2 measurements do not prove the intended 26.4 image. |

The existing shader build already copies source-hash-checked committed
metallibs on **any non-macOS host** (`reims-vgpu-vulkan/build.rs::main`).
That part does not need a new Windows Metal compiler or downloaded Apple
shader binaries. Changes to `.metal` sources still need the repository's
Mac compilation/recording workflow. The firmware's clang
`x86_64-unknown-windows` target is the EFI image target, not evidence that
the host library supports Windows.

Per [AGENTS.md](../AGENTS.md), documentation alone needs no build checks.
The implementing/merging work owns the checks its code changes reach;
GPU work must not overlap guest boots. This audit changes no source,
expected files, pins, guest disk or GPU configuration.

## Apple Tahoe license applicability

The official [macOS Tahoe 26 software license](https://www.apple.com/legal/sla/docs/macOSTahoe.pdf)
states in section 2J that its grants do not permit users to
"install, use or run the Apple Software on any non-Apple-branded computer".

Section 2A describes a single copy on one Apple-branded computer. For a
Mac App Store/automatic-download license, section 2B(iii) permits up to
two additional virtual instances on each Apple-branded computer owned or
controlled by the licensee that is already running the Apple Software,
for software development, development testing, macOS Server, or personal
noncommercial use. Section 2D also states Apple-branded hardware meeting
Apple's system requirements. Volume/maintenance terms and separate written
Apple agreements are addressed separately in section 2C.

These are the published terms, not a legal determination about this setup.
Apple's [license index](https://www.apple.com/legal/sla/) notes that the
license accompanying the product may differ from the version online;
the license accompanying the particular installer and any separate Apple
agreement must therefore be identified for applicability.

## Local evidence and audit validation

Uncommitted evidence is under `.work/diagnostics/audit/`: full REST
responses for each requested item and its comments, PR metadata/changed
files, the paginated #4 file list, fork branch resolutions, and the
official Tahoe PDF with retrieval metadata and SHA-256. No Apple binaries,
guest shader extracts, guest descriptors or logs are included in this
documentation commit.

Early implementation blockers were saved to `main-handoff.txt` in that
directory for the concurrent implementation's handoff.

Validation of this audit is source/commit comparison, live GitHub states,
official license text and a documentation diff check. There are no native
build, GPU-test, boot, performance or stability results from this task.
