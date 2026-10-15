# Continue from Tahoe recovery

1. WHPX is enabled and the host has restarted. Run start.ps1 to boot and
   qmp.ps1 for JSON commands, key/mouse input and guest screendumps. The VM
   uses WHPX, 8 vCPUs, 8 GiB, the validated OpenCore/OVMF/ROM, native Vulkan
   and the retained installer. Its enforced Haswell CPU masks ARAT.
2. OpenPartitionDxe detects the APM/HFS+ installer. Choose its observed
   OpenCore menu entry with QMP send-key. The picker waits for selection.
   The core/thread count MSR matches eight CPUs; recovery reaches its GUI.
3. The user accepted the agreement, and stage 1 copied its payload. The
   offline install-stage1 snapshot preserves that disk. Continue from the
   macOS Installer entry on the target, not the original media entry.
   Warm reset is fixed and verified with guest-requested reboots in the
   same process. RELEASE OpenCore with x86legacy passes the user-space IMG4
   board/digest checks that j185 failed, completes sealing and bless, and
   exposes the Reims boot entry. The install-sealed-x86legacy snapshot
   preserves it. Boot Reims and investigate XNU clock calibration. Preserve existing checkpoints when
   changing Secure Boot personalization.
   Use the account
   and region supplied by the owner; keep passwords out of source files,
   command lines and logs.
   Preserve SIP/AMFI settings and use the overlay chain.
4. Setup Assistant renders through AppleParavirtGPU/Vulkan. Complete account
   setup, then validate the installed Tahoe desktop and restart, AppleParavirtGPU/Metal,
   Finder/Dock/Safari, WindowServer/kernel logs and Liquid Glass screenshots.
   The guest repeatedly reports iconservicesagent crashes. Inspect its
   diagnostic reports, apply the documented METAL_ERROR_MODE setting and
   compare crashes, icon rendering and latency on this guest.
5. Use an active animation workload with benchmark.ps1; static or paused
   present counts are not FPS evidence. Compare exact rendering with
   REIMS_VGPU_EXACT=all, resolutions and cold/warm cache conditions. Add
   hardware timestamp and shader/cache/lock profiling before claiming GPU
   render duration or 1% low performance.
6. Test guest-page release/reuse, sustained GPU memory use and crash/reset
   recovery on the actual guest. Keep the remaining 32 GiB Vec case and full
   GPU/Mac suite limitations explicit until their checks can run correctly.

See windows-vm.md for stop, snapshot create/restore, log collection and
reproducible preparation. Current private artifacts and prepared snapshot
are retained under vm/. All three owned forks use windows-native-tahoe;
master and older branches are preserved.
