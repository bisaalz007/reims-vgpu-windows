# Native Windows validation

| Check | Result |
|---|---|
| Process-only PowerShell/MSYS2 artifact paths | Passed |
| Native C executable / EFI ROM format assertions | Passed |
| Renderer release and QEMU 11.1.50 builds | Passed |
| C header regeneration / drift check | Passed |
| Windows library clippy with warnings denied | Passed |
| xtask unit tests | 37 passed |
| Core CPU suite, serial | 2301 passed, 1 memory-heavy case filtered |
| Original unfiltered core run | 32 GiB Vec allocation abort; not green |
| C API unit suite | 46 passed, 1 child fixture ignored by its parent harness |
| Public API unit suite | 41 passed |
| Window/input/state unit suite | 91 passed |
| Windows VM/clock/thread/import tests | 6 passed |
| GPU backing/alias/lifetime cases | 12 passed |
| Native NVIDIA authored AIR pipeline | 1 passed |
| Native NVIDIA buffer copy/fill | 1 passed |
| metal2vulkan focused checks | 10 host-memory CPU, 1 CRT-exit, 1 native GPU passed |
| Firmware/VM/QMP/snapshot self-checks | 72 assertions passed, including retained-process power cycles, paused power-on and changed-option refusal |
| PCI/RAM realization, reset, QMP quit and native Win32 window | Passed in paused TCG diagnostic |
| Missing WHPX refusal | Passed; no fallback or guest boot |
| Real WHPX firmware boot and CLI status/key/mouse/screen commands | Passed; booter reaches handoff to XNU |
| WHPX core/thread count MSR | Six package cases passed; original code fails the test |
| Tahoe 26.4 recovery UI, sw_vers, SIP | Passed on eight CPUs; 25E246 and SIP enabled |
| Target GPT/APFS format and offline snapshot | Passed on the new 128 GiB overlay |
| Recovery shutdown | Guest Terminal command passed; QMP powerdown timed out |
| Installation stage 1 | Copied about 17.76 GiB and requested restart |
| WHPX warm restart | Native XSAVE/MTRR regressions passed; QMP reset and installer-requested reboot returned to OpenCore in the same process |
| Installation sealing | x86legacy passed board/digest verification, sealing and bless; guest reported success and rebooted to the Reims entry |
| Retained process power-off/on | OpenCore guest Shutdown reached QMP shutdown; CLI start resumed the same PID |
| Guest clock calibration | XNU 6.4 GHz disagrees with hypervisor/native host about 3.187 GHz |
| Installed Tahoe Setup Assistant | Panels render through AppleParavirtGPU and native NVIDIA Vulkan; account setup ongoing |
| Guest icon service / GPU stalls | Repeated iconservicesagent crashes and guest GPU hang reports; no host fatal in this run |
| Desktop/guest Metal inventory/Liquid Glass/workload FPS | Pending setup and validation |

NVIDIA evidence is for RTX 4070 SUPER (10de:2783), driver 616.56. Import
alignment is 4096 bytes. Private and shared-section memory, including two
reversed 4 KiB placeholder views, passed bidirectional CPU/GPU assertions.
Direct read-only imports were refused by the driver; the runtime now returns
a typed HostMemory refusal and Reims creates writable twins. Twin CPU tests
and the integrated backing cases passed.

A native blank-frame window opened and presented while guest CPUs were
paused. Its low present rate is not a desktop performance measurement.
The original installer is untouched; file metadata/hash and evidence logs
are in ignored vm/state, vm/logs and .work/diagnostics.
