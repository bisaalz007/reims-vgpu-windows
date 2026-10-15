# Windows native Tahoe implementation

Windows Hypervisor Platform is enabled after the host restart. A real WHPX
boot reaches the installed Tahoe 26.4 Setup Assistant after successful
sealing and bless. The new 128 GiB target is GPT/APFS, and recovery reports
SIP enabled. Account setup and desktop validation are ongoing; this is not
a Liquid Glass correctness or performance claim.

## Sources and toolchain

- Reims base: steelbrain/reims-vgpu `9b2c5452de94a474cba28d80d7f99d9d7b19ffe3`.
- QEMU Windows branch: tqmane/qemu-reims-vgpu `45807f2484d97db11943287d85fdee37e7ff991a`,
  based on `1a72df984914fb9929d35b1217619178133d65a4`.
- metal2vulkan Windows branch: tqmane/metal2vulkan `8556e6e4fe54fd8edcf45d48415242a2ac875421`,
  based on `13b279df1bf87893d389f955a90c55b6ff3d0f9b`.
- Native GNU Rust 1.98.1, MSYS2 UCRT64 Clang/LLVM/LLD 22.1.8,
  CMake 4.4.0, Meson 1.11.2, Ninja 1.13.2, Python 3.14.6 and cbindgen 0.29.4.
  All project caches and artifacts stay under the project. Existing MSYS2
  packages were updated; the before/after inventory is in private diagnostics.

## Implemented and verified

Windows guest RAM uses real retained section handles and page-aligned
MapViewOfFile3 aliases. The additive C config accepts old 48/56-byte callers;
its new sections field is at offset 56. Core clocks, threads, death tests,
IOSurface stand-ins, SDL3 Win32 windows and native Vulkan swapchains compile.
Read-only CPU views use writable import twins, with private zeros and import
lifetime retention. The runtime validates Windows pages before driver import.
The native QEMU has WHPX, the current apple-gfx-pci ABI, selective WHPX fixes,
correct q35 dirty-sync offsets and the current lockless BAR path.

The optimized Reims DLL and QEMU 11.1.50 build and execute on Windows.
UCRT64 must precede Rust's bundled DLLs on PATH; the environment helper does
this. Unused Curses, GnuTLS and Nettle backends are disabled: the existing
host libraries could not load, and Nettle 4 has an incompatible API. SDL,
localhost QMP and user networking remain enabled.

The supplied media was identified as APM/HFS+, macOS 26.4 build 25E246, with
an x86_64 PE booter. Its original was not changed. Exact file details and
SHA-256 are private in vm/state/media.json.

OpenCore 1.0.8 is from its official release and passes ocvalidate. The FAT
EFI image, matched QEMU OVMF pair, native-built Reims option ROM and 128 GiB
sparse qcow2 overlay chain are prepared. iMac20,1/x86legacy is configured; CSR is
zero, signed DMG loading is required, and no AMFI disabling argument is set.
Sample identity values do not provision Apple online services.

A paused TCG diagnostic realized PCI 106b:eeee with 8 vCPUs and 8 GiB RAM.
PCI/memory QMP queries, device reset, normal QMP quit and a second start/stop
worked. A native window titled Reims vGPU opened at 1920x1080 and presented
through MAILBOX on the Vulkan swapchain. This diagnostic never ran a guest
CPU. Normal start refuses unavailable WHPX and does not fall back to TCG.
The prepared snapshot is an empty install baseline, not an installed macOS.

After restart, WHvGetCapability and Win32_ComputerSystem both report a
running hypervisor. The enforced Haswell CPU string masks only ARAT, which
WHPX does not expose on this host, and uses a supported Haswell stepping 3.
MSR_CORE_THREAD_COUNT now uses QEMU's existing topology helper, matching the
eight ACPI processors. OpenPartitionDxe enables detection of the
APM/HFS+ installer. QMP CLI status, key input, absolute mouse events and guest
screendumps work on the actual WHPX VM. Recovery's sw_vers confirms 26.4,
25E246; csrutil reports enabled. The verified empty 128 GiB overlay is
GPT/APFS, with a Reims volume. Guest-command shutdown worked, and an offline
before-install snapshot was created. Recovery ignored QMP powerdown within
15 seconds; installed-system shutdown still needs verification.

The selected RTX 4070 SUPER, Windows driver 616.56, passed native host-import,
CPU/GPU copy/fill and authored Metal AIR compute-pipeline tests. See
TEST_RESULTS.md for scope and KNOWN_ISSUES.md for remaining checks.

## Continuation

Use scripts/windows/start.ps1 and qmp.ps1 for CLI operation. Diagnose the
installer from serial output and guest screendumps. Only the user may
accept the installer agreement. Use the account and region supplied by the
owner; keep passwords out of source files, command lines and logs.
The accompanying Apple license must be considered; see
windows-vm.md and windows-native-audit.md.

The user accepted the installer agreement. Stage 1 copied about 17.76 GiB
and restarted, exposing a WHPX XSAVE/warm-reset failure. Native XSAVE
buffer negotiation and reset of extended, interrupt and MTRR state now
allow both QMP and installer-requested reboots in the same process. Disk
checkpoints preserve stages 1 and 2. Sealing initially failed MSU 1130:
user-space IMG4 selected VM board f0 against a j185 manifest with board 22.
The kernel ForceSecureBootScheme patch was confirmed applied but did not
resolve that user-space mismatch. RELEASE OpenCore is restored, and the
VM's documented x86legacy model passes board and digest verification.
Sealing and bless finished successfully, and an automatic restart exposed
the Reims boot entry. An offline install-sealed-x86legacy checkpoint
preserves that state. OpenCore's guest Shutdown and CLI start also powered
off/on in the same QEMU process. The XNU/hypervisor clock-frequency discrepancy
remains unresolved.

Setup Assistant renders through AppleParavirtGPU and the native NVIDIA
Vulkan runtime. The guest repeatedly reports iconservicesagent crashes;
the documented METAL_ERROR_MODE workaround still needs validation on this
guest. Account setup, full desktop/reboot, guest Metal inventory, Liquid
Glass and workload FPS remain unverified. No 60/120 FPS result is claimed. Private evidence is in
.work/diagnostics/ and vm/logs/. Generated binaries, guests and logs are ignored.
