# AGENTS.md

Operating guide for AI agents working in this repository.

## What This Project Is

This research project emulates Apple's paravirtualized GPU on the host. An unmodified macOS guest
uses Apple's own GPU drivers; our QEMU device and Rust backend decode the command stream and
execute it through Metal or Vulkan. We ship no guest driver.

`crates/reims-vgpu` supports three first-class pathways:

| Pathway | Host | Guest | Attach | Page shift | Backend | Boot |
|---|---|---|---|---|---|---|
| x86 macOS / Linux Vulkan | Linux x86_64 (KVM) | x86_64 macOS Metal guest | PCI (`reims-vgpu-pci`) | 12 | Vulkan | `vm/boot-x86.sh` |
| arm64 macOS / macOS Metal | Apple Silicon macOS (HVF) | arm64 macOS Metal guest | sysbus MMIO (`reims-vgpu-mmio`) | 14 | Metal-direct | `vm/boot-arm64.sh` |
| arm64 macOS / macOS Vulkan | Apple Silicon macOS (HVF) | arm64 macOS Metal guest | sysbus MMIO (`reims-vgpu-mmio`) | 14 | Vulkan through MoltenVK | `vm/boot-arm64.sh` |

Pathway-specific facts must be verified on the pathway being changed. Do not generalize from arm64
to x86, from Metal to Vulkan, or from one host GPU class to another.

### A fourth pathway: the host boots macOS, the device is not proven on it

| Pathway | Host | Guest | Attach | Page shift | Backend | Boot |
|---|---|---|---|---|---|---|
| x86 macOS / Windows Vulkan | Windows x86_64 (**WHPX**) | x86_64 macOS Metal guest | PCI (`reims-vgpu-pci`) | 12 | Vulkan | `vm/boot-x86.sh` (`ACCEL=whpx`) |

Treat this as bring-up, not a supported rail. Measured on 2026-07-31 on a Windows 11 host
(RTX 4070 SUPER + Intel UHD 770, MSYS2 MINGW64), so do not re-derive it:

- **The whole toolchain builds and the crate passes on Windows.** 1122 tests, clippy
  `--target x86_64-pc-windows-gnu --all-targets -D warnings` clean, feature matrix green,
  `qemu-system-x86_64.exe` links with `reims-vgpu-pci` present and `whpx` available. The
  GPU-touching `vk_engine_batch` tests pass, so the Vulkan engine executes against a real ICD here.
- **x86_64 macOS Tahoe 26.4 boots under QEMU+WHPX to its installer UI** (Darwin 25.4.0,
  `xnu-12377.101.15`), with all 8 vCPUs enumerated. This is the claim that previously had no public
  report; it now has one here, on this tree. It was reached with `vmware-svga`, **not** with
  `reims-vgpu-pci` — see the open questions below.
- `-accel whpx,ignore-unknown-msr=` defaults **on** (`accel/whpx/whpx-common.c:557`,
  `target/i386/whpx/whpx-all.c:2922`), so the Linux rail's `options kvm ignore_msrs=1` requirement
  has **no Windows equivalent to configure**.
- **Keep the guest at 8 vCPUs.** WHPX stops the VM outright when a vCPU returns exit reason 4
  (`InvalidVpRegisterValue`): QEMU logs `whpx: cpu N stopped the VM: unexpected exit reason 4 at
  cs:rip ...` and the monitor then reports `VM status: paused`. Measured at `-smp 12,cores=6` on a
  logged-in desktop — four vCPUs hit it within minutes, at four different guest RIPs, with no guest
  panic and no message on the serial console. `-smp 8,cores=4` runs the same session without it. It
  presents as a total guest freeze while the host window still redraws and the drain worker still
  turns (`drain_duty ... draws=0 computes=0`), so it is easy to mistake for a device hang — check
  `info status` before suspecting the device. `-smp 16` was already known to fail this way during
  reset; the failure is not confined to reset.
- **macOS idles into suspend-to-RAM and does not come back.** The guest logs `IOCPUSleepKernel
  enter`, powers its devices down, and no keyboard or pointer event revives it; the screen stays
  black with the host process alive, which reads exactly like a render failure and is not one. Do
  not reach for `-global ICH9-LPC.disable_s3=1` to prevent it: measured here, taking S3 away made
  things worse rather than better — two sessions then stopped with `whpx: cpu N stopped the VM:
  unexpected exit reason 4`, once at four different kernel RIPs and once with four vCPUs at the
  identical low-memory address `0038:0000000000002012`, which is AP-startup/resume trampoline
  territory. Neither happened on the same tree with S3 left in place. Turn the idle timer off inside
  the guest instead (`sudo pmset -a sleep 0 displaysleep 0 disablesleep 1`).
- **CPUID leaves `0x2`, `0x4`, `0x5` are now in the WHPX exit list** and must stay there. WHPX lets
  the hypervisor answer any leaf it does not exit, and XNU takes cores-per-package straight from
  leaf 4 without consulting `-smp`. On a hybrid P/E-core host that produced a topology the guest
  could not use: `-smp 8` hung in early SMP bring-up with **no console output at all**, `-smp 1`
  booted. Adding the leaves fixed it. A silent early hang with the Apple logo still on screen is
  what this failure looks like — reach for CPUID before suspecting the device.

Three host facts that cost a build each, and are not discoverable from the code:

- **Windows Developer Mode must be on.** `meson setup` runs `symlink-install-tree.py`, whose
  `os.symlink` fails with `WinError 1314` without it, and meson reports only "Postconf script
  failed" with the real error swallowed.
- **`PYTHONUTF8=1` is required** and `scripts/qemu-build/qemu-build.sh` now exports it. QEMU's build
  scripts read sources with a bare `open()`, which takes its encoding from the system locale; on a
  Japanese Windows that is cp932 and `tracetool.py` dies on the first non-ASCII byte in
  `hw/display/trace-events`, 229 targets in.
- **`llvm-dis` and `spirv-val` must be on `PATH` at run time**, not just for tests — metal2vulkan
  shells out to the first for AIR translation, and `vm/boot-x86.sh` refuses to start without either.
  `mingw-w64-x86_64-llvm` and `mingw-w64-x86_64-spirv-tools` provide them.
- **`x86_64-unknown-uefi` cannot be added to a distribution rustc.** MSYS2's
  `mingw-w64-x86_64-rust` has no rustup, so `crates/reims-vgpu-efi` does not build there.
  `vm/boot-x86.sh` now skips the option ROM and says so rather than failing the boot — the ROM only
  provides a GOP before the product present path latches, and `REIMS_VGPU_GOP_ROM=` already existed
  to turn it off. Set `REIMS_VGPU_GOP_ROM` to any value to make that failure fatal again.

Guest-media facts, if a macOS installer image is involved:

- A "macOS ISO" is usually **not ISO9660**. The one measured here is a raw disk with an Apple
  Partition Map and a single `Apple_HFS` partition, and it must be attached as a disk, not CD media.
- **OVMF cannot read an Apple Partition Map.** EDK2's `PartitionDxe` handles MBR, GPT, El Torito and
  UDF; real Mac firmware handles APM and OVMF does not. The partition never becomes a block device,
  so OpenCore's picker shows only its own ESP.
- **Stripping the map instead is a trap.** XNU roots by boot-uuid (`rooting via boot-uuid from
  /chosen`) and `AppleFileSystemDriver` publishes that UUID from a *partition* `IOMedia`. A disk with
  no partition map has nothing to match, and the kernel sits at `Still waiting for root device`
  with every disk detected and powered up. **A GPT describing the same extent satisfies both ends**;
  nothing needs to be copied, only the first and last sectors rewritten.
- OpenCore classes installer volumes as **auxiliary** entries. With `HideAuxiliary=true` the picker
  hides them until Space is pressed, which reads exactly like the volume not being found.
- Set `boot-args` to include `-v serial=1` before diagnosing anything: the kernel log then lands in
  the `-serial` file as text instead of only on a screen that another window can overlap.

Still untested and dangerous to assume: whether `snapshots/current` behaves as a junction under
MSYS2, and whether AppleParavirtGPU issues any vector instruction to BAR0 MMIO (which QEMU's
in-tree emulator cannot decode — it would `abort()` on an unknown opcode, or silently skip a
decoded-but-unhandled one). QEMU's **AF_UNIX QMP socket is still unverified here**; the monitor over
`tcp:127.0.0.1:PORT` does work and is the safe choice on this host.

Nothing above says the device works. `reims-vgpu-pci` realizes under WHPX, the Vulkan backend
initialises, and the always-on sinks are written — but no guest has attached to it, so every
question about MMIO protocol, opcode coverage and present remains open on this rail.

The intended guest for this pathway is macOS 26 Tahoe, and the guest-driver question is **settled
in the affirmative**. Measured on an Intel Tahoe install (2026-07-31), so do not re-derive it:

| Fact | Value |
|---|---|
| `AppleParavirtGPU.kext/Contents/Info.plist` → `IOPCIMatch` | `0xEEEE106B` — i.e. vendor `0x106B`, device `0xEEEE` |
| `IOProviderClass` | `IOPCIDevice` |
| Kernel code | present in `/System/Library/KernelCollections/SystemKernelExtensions.kc` as an `LC_FILESET_ENTRY` for `com.apple.driver.AppleParavirtGPU`; `__TEXT,__text` 77626 bytes, `LC_SOURCE_VERSION` 63.4.3, UUID `1A14B80D-E04B-3199-8F6C-55D735ADD7E3` |
| `AppleParavirtGPUMetal.bundle` | universal, **x86_64** + arm64e |
| `AppleParavirtGPUMetalIOGPUFamily.bundle` | universal, **x86_64** + arm64e |

So `hw/display/reims-vgpu-pci.c`'s `0x106B:0xEEEE` is still the right pairing on Tahoe, the x86_64
user-space Metal stack was not dropped, and the kext is not a codeless stub. Four hypotheses are
dead and must not be re-raised without new evidence: that Tahoe changed the PCI ID, that it removed
the paravirt GPU from Intel builds, that the Metal driver went arm64e-only, and that the kext is
metadata alone.

#### Tahoe ships *two* paravirt GPU stacks, and only one can bind over PCI

Read off the shipped `IOKitPersonalities` on macOS 26.3.1 build 25D771280a (x86_64):

| Kext | Personality | Provider | Match | Metal plugin |
|---|---|---|---|---|
| `AppleParavirtGPU` 15.0.0 | `AppleParavirtGPUControl` | `IOPCIDevice` | `IOPCIMatch 0xEEEE106B`, `IOProbeScore 1000000` | `AppleParavirtGPUMetal` / `AppleParavirtDevice` |
| `AppleParavirtGPUIOGPUFamily` 15.0.0 | `AppleParavirtGPUARM` | `AppleARMIODevice` | `IONameMatch paravirtualizedgraphics,gpu` | `AppleParavirtGPUMetalIOGPUFamily` / `AppleParavirtDevice` |
| `AppleParavirtIOSurface` 15.0.0 | `ParavirtMapperDevice` | `AppleARMIODevice` | `IONameMatch paravirtualizedgraphics,iosurface` | — |

The two stacks **do not compete**: the IOGPUFamily one matches a device-tree name on an
`AppleARMIODevice`, which does not exist on a PCI x86 machine. So an x86 guest attached to
`reims-vgpu-pci` binds the legacy IOAccelerator-family stack, and only that one — the same stack the
crate's decode was built against. There is no probe contest to lose, and `IOProbeScore 1000000`
puts it far above any generic display driver matching the same ID.

This is also the line the repo's two attach shims fall along: `reims-vgpu-pci` (x86, page shift 12)
faces `AppleParavirtGPUControl`; `reims-vgpu-mmio` (arm64 `vmapple`, page shift 14) faces the
`AppleARMIODevice` side. Do not reason from one to the other.

Corroborating: only `com.apple.driver.AppleParavirtGPU` has an `LC_FILESET_ENTRY` in the x86
`SystemKernelExtensions.kc`; the other two have none. All 1118 `AppleParavirt*` symbols in that
collection belong to the PCI stack — the class `AppleParavirtGPU` (the IOGPUFamily `IOClass`) has
zero symbols there.

#### The PCI stack's interface surface, as shipped in Tahoe

Class map (method counts from the kernel collection's symbol table), useful for orienting decode
work against real driver structure rather than guesses:

`AppleParavirtResource` 107 · `AppleParavirtPageTable` 106 · `AppleParavirtDisplayPipe` 101 ·
`AppleParavirtFramebuffer` 78 · `AppleParavirtAccelerator` 76 · `AppleParavirtShared` 50 ·
`AppleParavirtVirtualChannel` 41 · `AppleParavirtResourceHeap` 39 · `AppleParavirtCommandQueue` 37 ·
`AppleParavirtSharedUserClient` 34 · `AppleParavirtResourceHeapNamespace` 33 · `AppleParavirtTask` 32 ·
`AppleParavirtSegmentResourceList` 30 · `AppleParavirtDisplayMachine` 30 · `AppleParavirtEventMachine` 28 ·
`AppleParavirtGPUControl` 27 · `AppleParavirt2DContext` 26 · `AppleParavirtRootChannel` 24 ·
`AppleParavirtMemoryMap` 24 · `AppleParavirtKernelTask` 23 · `AppleParavirtVidMemory` 22 ·
`AppleParavirtDisplayPipeFence` 21 · `AppleParavirtResourceHeapHandle` 20 · `AppleParavirtSysMemory` 19 ·
`AppleParavirtChannel` 19 · `AppleParavirtCommandAllocator` 12

Names that matter for bring-up:

- `AppleParavirtGPUControl::start` → `::setupMMIO` is the PCI attach entry point. **The first BAR0
  access from the guest OS comes from `setupMMIO`.** Anything touching BAR0 before that is firmware
  (OVMF/the option ROM), so an attach witness that cannot tell them apart proves nothing.
- `AppleParavirtAccelerator::configureDevice(IOPCIDevice*)`, `::setupChannels`, `::setupVersion`,
  `::newCommandQueue`, `::newDisplayPipe`, `::getStampMemory` — the bring-up sequence, in the order
  the log should show it.
- `AppleParavirtRootChannel::submitBuffer` and `AppleParavirtVirtualChannel::submitBuffer` — the two
  submission paths behind the root/child opcode split the crate decodes.
- `AppleParavirtCommandQueue::parseSegmentList`, `::processSegmentKernelCommand`,
  `::processExecIndirect`, `::writeInvalidates` — the command-stream shape, including the indirect
  path the crate handles in `runtime/icb/`.
- `AppleParavirtChannel::getChannelLabel` — channels are labelled in the driver, so channel identity
  is a driver-side concept, not a number this project invented.
- `AppleParavirtRootChannel::resetHardwareAndReplay` / `::hasFault` /
  `::getHardwareDiagnosisReport` — the guest **can** decide the device faulted and replay. A boot
  that silently loops may be the guest resetting us, not us hanging.

Versions to diff against a Ventura 13 install when asking "did the protocol drift": kext
`CFBundleVersion` 15.0.0, `LC_SOURCE_VERSION` 63.4.3, Metal bundles `CFBundleVersion` 63.4.3.

Two things this does **not** establish, because they cannot be seen from the file system: whether
Tahoe's MMIO/command-stream protocol still matches what this crate decodes, and whether Tahoe-era
`.mtlb`/AIR is within metal2vulkan's coverage. `LC_SOURCE_VERSION` 63.4.3 is the cheapest available
proxy for protocol drift — compare it against the same field on a Ventura 13 install before
assuming the decoder is current.

`is_tahoe` (`runtime/drain/mod.rs`) is presently **dead state**: written on
`ROOT_OP_DEVICE_INFO_TAHOE` (`0x3a`) and read only by a test. Every dated live-RE annotation on the
x86 rail names Ventura 13.7.8. Do not read "Tahoe is decoded" into the presence of that opcode.

## Main Components

- `vendor/qemu` - QEMU fork with the thin device shim: QOM, MMIO/BAR, IRQ/MSI, console/display
  integration, and HostOps plumbing.
- `crates/reims-vgpu` - Rust staticlib that owns protocol decode, device model, memory mapping,
  command planning/execution, scheduling, and Metal/Vulkan backend behavior.
- `crates/reims-vgpu/src/observe/` - crate-wide observability: fail logs, typed decline reasons,
  emission helpers, and gates.
- `vm/` - snapshot-revert boot scripts for arm64 and x86 guests.

Start with the owning source modules and nearby tests when changing device, decode, present, or
backend behavior. Keep durable design facts in tracked docs or code comments close to the behavior
they explain.

## Operating Principles

### C Is A Thin Shim

C and Objective-C in the QEMU path exist to connect QEMU to Rust. Keep product logic in
`crates/reims-vgpu`: protocol interpretation, resource state, scheduling, GPU encode, present model,
backend policy, and performance behavior belong in Rust.

### Never Fail Silently

If a decoded guest command is rejected, dropped, degraded, unsupported, or mis-executed, make the
reason visible. Use typed decline/refusal reasons and emit them through the always-on failure path
so `/tmp/reims-vgpu-fail.log` explains what happened.

That path is `/tmp` **literally** on the Unix rails — this doc names it, the boot scripts open it,
and offline analysis greps it, so the sink does not follow `TMPDIR`. On Windows there is no `/tmp`
(the spelling is drive-relative and would drop every line), so the sink falls back to the OS temp
directory. `REIMS_VGPU_LOG_DIR` overrides on every host. One helper decides this for the sinks and
for every other artifact the crate drops — the GOP console proxy, the compute-stall SPIR-V dump, the
metal2vulkan handoff last resort — so a new host is one edit in `observe::sink`, not a literal
repeated across `runtime/`. Do not reintroduce a bare `"/tmp/..."` path anywhere in the crate.

Expected control flow should stay quiet. A resolver saying "not ready yet" or an intentionally
unbound `ref == 0` is not a failure. A real loss of guest work is.

### Measure Before Fixing

You cannot fix what you cannot measure. If we do not know what class of failure we are fixing, we
are operating blind and guessing.

Before landing a visual, protocol, performance, or translation fix, add or identify a log-level or
test-level proxy for the bug class. Screenshots are useful evidence, but they are not a regression
gate by themselves.

### Tests Define Done

If there is no test for it, it is not done. No test means a future agent can regress the changeset
without noticing.

Behavior changes need tests that fail without the change. Bug fixes need a focused synthetic case
or proxy test for the bug class. Run Rust tests serially with `-- --test-threads=1`; GPU-touching
tests are not safe to run in parallel.

### Do Not Overfit Fixes

Never special-case behavior for a screenshot, boot stage, pixel dimension, resource size, object id,
function name, pipeline ref, or observed content pattern. Implement the decoded API contract.

Temporary probes are fine when they collect evidence. Remove probe-only behavior before claiming the
fix. Do not turn observations into product heuristics.

### No Magic Numbers

Do not guess numbers because they fit one observation. Derive constants from the contract: SDK
headers, `sizeof`/`offsetof`, decoded guest fields, documented serializer output, or controlled
empirical measurement. Record the basis in the code, tracked docs, or the commit body when the
value is not obvious.

Guest page geometry is always explicit. Portable code takes `page_shift` or `page_size`; arch-fixed
helpers must say so in their names.

### Write Comments For The Code

Inline comments should explain the code as it exists now. Do not mention the prompt, a temporary
plan, a phase number, a section of an implementation plan, or the reason a plan asked for the
change. Plans are consumed and deleted; code comments outlive them and must not point at things
that no longer exist.

### Keep Claims Narrow

State exactly what you verified. A single green boot does not prove an entire class is fixed. Broad
claims such as "zero-copy everywhere" or "no fallback remains" require an audit of every place that
could falsify them.

## Support Matrix

arm64 and x86 are both first-class. Metal and Vulkan are both first-class where the host supports
them.

The Vulkan backend must support all four memory/DMA cells:

| | DMA available | No DMA available |
|---|---|---|
| Unified memory | Apple M-series / MoltenVK, Intel/AMD iGPU on Mesa | Unified-memory hosts with no sharing mechanism |
| Discrete memory | Discrete GPUs with a working sharing path | Discrete GPUs that require copy crossings |

Vulkan 1.2 is the baseline. Anything above Vulkan 1.2 must have a fallback or a capability-gated
path. Gate on capabilities, not vendor names, driver names, or API-version assumptions.

Host-pointer imports must stay windowed. Do not import the whole guest RAM VMA for GPU DMA; it pins
host RAM. Use the existing capped window resolver.

## Verification

Pick the pathway your change affects.

- Arm64: `vm/boot-arm64.sh --device reims-vgpu-mmio --testing`, then
  `scripts/screenshot-when-macos-host/screenshot-when-macos-host.sh /tmp/screen.png`
- x86: `vm/boot-x86.sh --device reims-vgpu-pci --testing`, then
  `scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh -o /tmp/screen.png`
- x86 on Windows (device unproven): from a **MINGW64** MSYS2 shell,
  `vm/boot-x86.sh --device reims-vgpu-pci --testing` (picks `ACCEL=whpx` from `uname -s`), then
  `scripts/screenshot-when-windows-host/screenshot-when-windows-host.sh -o /tmp/screen.png`.
  That script captures with `PrintWindow`, not a screen copy, so a window overlapping the VM cannot
  end up in the "screenshot" — which happened on the first capture taken on this rail and is
  indistinguishable from guest output. Other evidence is `$RUN_DIR/serial-*.log`,
  `$RUN_DIR/qemu-*.log` (QEMU's own `-D`, which carries the `-cpu ...,check` feature-strip warnings
  — the ground truth for what WHPX actually exposes) and the fail sink under `%TEMP%` or
  `REIMS_VGPU_LOG_DIR`.

  To create the guest in the first place: `vm/boot-x86.sh --installer <image>`. It creates the
  target disk, wraps an Apple-Partition-Map installer in a GPT without copying or modifying it, and
  boots it. Put `-v serial=1` in OpenCore's `boot-args` before diagnosing anything the guest does —
  the kernel log then lands in `$RUN_DIR/serial-*.log` as text.

For Rust changes, run the relevant native tests serially from the repo root:

```sh
cargo test -p reims-vgpu --no-default-features --features backend-vulkan,host-window -- --test-threads=1
cargo test -p reims-vgpu --no-default-features --features backend-metal -- --test-threads=1
```

`backend-metal` is Apple-only; run that arm only on Apple hosts. Run the feature matrix from the
repo root when cfgs, features, backend boundaries, or shared Rust code change:

```sh
scripts/feature-matrix/feature-matrix.sh
```

Before and after long Rust test runs, sweep orphaned test binaries:

```sh
pkill -9 -f 'target/debug/deps/reims_vgp[u]-'
```

On Windows there is no `pkill`, and a live test binary holds a lock that makes the next `cargo
build` fail with a confusing "Access is denied" on the executable rather than a test failure:

```sh
taskkill //F //IM 'reims_vgpu-*.exe' 2>/dev/null || true
```

### Vulkan validation layers without root

Sync validation (`SYNC-HAZARD-*`) names a read/write hazard at the command that causes
it. Five such hazards were found and fixed by hand on this rail before anyone tried it;
every one would have been a single log line. The device creates its instance with no
layers, so the loader's environment is the whole mechanism — and the layer package does
not have to be installed system-wide, which is what previously blocked this behind a
`sudo` password:

```sh
mkdir -p /tmp/vklayers && cd /tmp/vklayers
curl -sO https://geo.mirror.pkgbuild.com/extra/os/x86_64/vulkan-validation-layers-<ver>-x86_64.pkg.tar.zst
tar --use-compress-program=unzstd -xf vulkan-validation-layers-*.pkg.tar.zst
```

The layer manifest names its library relatively, so the extracted `usr/lib` has to be on
the loader's search path:

```sh
export VK_LAYER_PATH=/tmp/vklayers/usr/share/vulkan/explicit_layer.d
export LD_LIBRARY_PATH=/tmp/vklayers/usr/lib
export VK_LOADER_LAYERS_ENABLE='*validation*'
export VK_LAYER_SETTINGS_PATH=/tmp/vklayers/vk_layer_settings.txt
```

`vk_layer_settings.txt` is where sync validation is turned on and where the output goes
somewhere a repro can slice:

```text
khronos_validation.validate_sync = true
khronos_validation.report_flags = error,warn
khronos_validation.log_filename = /tmp/reims-vgpu-vkvalidation.log
khronos_validation.duplicate_message_limit = 3
```

Confirm the layer actually loaded before believing a clean run — `vulkaninfo --summary`
must list `VK_LAYER_KHRONOS_validation` under `Instance Layers`. A boot with sync
validation on is not a frame-rate or timing measurement.

## Commit Guidelines

Commit only work you wrote. Never commit third-party code or intellectual property, including Apple
software, firmware, disk images, `.mtlb`, AIR, or SPIR-V. Keep those artifacts ignored and local.
Reports may include original analysis, metadata, hashes, and reproduction steps, but no third-party
bytes or excerpts.

Each commit should have a detailed message body that states:

- Which component or pathway it touches.
- What behavior changed and why.
- What tests, clippy runs, feature-matrix checks, or live-VM verification were performed.
- What was not verified, if anything.

Rust commits should be warning-free under clippy with `-D warnings` for every affected matrix arm.
Use the appropriate subset for the host and change; the Metal command is Apple-only:

```sh
cargo clippy -p reims-vgpu --all-targets --features backend-metal -- -D warnings
cargo clippy -p reims-vgpu --all-targets --no-default-features --features backend-vulkan,host-window -- -D warnings
cargo clippy -p reims-vgpu --target x86_64-unknown-linux-gnu --all-targets --no-default-features --features backend-vulkan,host-window -- -D warnings
cargo clippy -p reims-vgpu --target x86_64-pc-windows-gnu --all-targets --no-default-features --features backend-vulkan,host-window -- -D warnings
```

The Windows arm needs `rustup target add x86_64-pc-windows-gnu` and is skipped-and-named by the
feature matrix when absent. Run it whenever you touch a `target_os` gate, `observe::sink`, the
`host_window` surface/event-loop code, or `hw/display/meson.build`.

Do not hide warnings, skip an affected arm, or commit a dropped test count without calling it out.
