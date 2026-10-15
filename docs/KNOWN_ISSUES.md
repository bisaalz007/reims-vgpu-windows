# Known issues

- WHPX boots the Tahoe 26.4 recovery UI with eight CPUs after correcting
  Haswell stepping and the core/thread count MSR. GPT/APFS preparation is
  complete. The user accepted the agreement; installation stage 1 copied
  approximately 17.76 GiB and reached its automatic restart.
- WHPX warm reset now negotiates the native XSAVE buffer size and resets
  extended, interrupt and MTRR state. QMP reset and an installer-requested
  reboot returned to OpenCore in the same process. Retained-process power
  cycles pass the QMP fixture and OpenCore guest firmware power-off/on.
  Installed-macOS power-off/on still needs proof.
- The j185 configuration failed sealing with MSU 1130: its manifest board
  22 disagreed with the user-space VM verifier's board f0. Kernel
  ForceSecureBootScheme alone did not resolve it. RELEASE OpenCore with
  x86legacy passes board/digest verification, completes sealing and bless,
  and exposes the installed Reims boot entry after automatic restart.
  SIP/AMFI and signed-DMG policy remain enabled.
- A discarded per-vCPU delete/recreate reset experiment preceded a host
  HYPERVISOR_ERROR bugcheck. It is excluded from the current source. The
  disk passed an offline qcow2 check; an exact kernel cause is unproven.
- XNU reported RTCLOCK 6.4 GHz, while the Windows hypervisor capability and
  native RDTSC/QPC measurement report about 3.187 GHz. Clock calibration is
  being investigated; installer remaining-time and performance claims need
  this discrepancy resolved.
- Recovery ignored QMP system_powerdown within 15 seconds. Its Terminal
  shutdown command exited cleanly; installed-system shutdown is unverified.
- Installed Tahoe reaches Setup Assistant and renders its panels through
  AppleParavirtGPU/Vulkan. During that run, iconservicesagent crashes
  repeatedly and the guest reports GPU hangs. No host fatal was logged.
  Validate the documented METAL_ERROR_MODE setting, crash reports and an
  active workload before concluding which delays it resolves.
- Account setup, desktop/reboot, Finder/Dock/Safari, Liquid Glass and
  sustained FPS remain unverified.
- One legacy core test, tasks::shared::tests::data_from_virtual_offset::
  a_length_past_any_task_raises, requires a normal 32 GiB Vec allocation.
  It aborts on the current Windows memory/commit limit and remains unchanged.
  The other 2301 core tests passed serially. Large fake raw regions use
  sparse temporary files, preserving their zero-filled real address space.
- No macOS behavior-lock suite or Mac runtime check was run. Linux shared
  sources were type-checked on the Windows host; this is not a Linux execution
  claim or a substitute for Windows guest evidence.
- Window GPU statistics currently use the runtime's host-clock interval,
  which includes submission/completion latency. They are not pure hardware
  GPU start/end timestamps. Absolute Metal clock equivalence, calibration,
  shader/cache/lock profiling and 1% low workload measurement remain open.
- Native copy/fill/pipeline and QMP shutdown passed. Full GPU stress/device
  loss, DLL unload under loader lock, mixed-CRT shutdown under guest load,
  long-running resource reclamation and guest-page reuse still need checks.
- Runtime labels the Windows NVIDIA driver identity unmeasured and selects
  its exact paths. These paths are not proof of Tahoe UI rendering correctness.
- NVIDIA/Windows internal driver and OS caches are not controlled by the
  project environment. Explicit project data remains under the project.
