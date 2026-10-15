//! Installing macOS from an IPSW into a new arm64 VM, through Virtualization.framework, the only
//! thing that restores one. The result is what macosvm makes and QEMU's vmapple boots: a raw
//! `disk.img`, the auxiliary storage `aux.img`, and a machine identifier, whose ECID
//! `-M vmapple,uuid=` takes (the identifier's and the hardware model's data are kept beside them,
//! in `machine-identifier.bin` and `hardware-model.bin`).
//!
//! The install runs in a child process (`reims-ctl __vz-install`): Virtualization.framework uses
//! a virtual machine on its queue, the main one here, which the child's main thread serves by
//! running the main run loop, whatever the parent's threads are doing (the interface runs
//! installs on a thread of its own). The child prints `progress <fraction>` lines and, at the
//! end, `ecid <n>`. It needs the `com.apple.security.virtualization` entitlement, which
//! scripts/install-ctl.sh signs reims-ctl with.

use anyhow::{Context, Result, bail};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Whether this binary carries the virtualization entitlement.
pub fn entitled() -> bool {
    if !cfg!(target_os = "macos") {
        return false;
    }
    let Ok(exe) = std::env::current_exe() else {
        return false;
    };
    Command::new("codesign")
        .args(["-d", "--entitlements", "-", "--xml"])
        .arg(&exe)
        .stderr(Stdio::null())
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("com.apple.security.virtualization"))
        .unwrap_or(false)
}

/// Installs `ipsw` (a path, or `latest`) into `dir`, making a disk of `disk_mib`; returns the
/// machine identifier's ECID.
pub fn install(
    ipsw: &str,
    dir: &Path,
    disk_mib: u64,
    cpus: u32,
    memory_mib: u64,
    progress: &mut dyn FnMut(&str),
) -> Result<u64> {
    if !cfg!(target_os = "macos") {
        bail!("an IPSW installs only on macOS");
    }
    if !entitled() {
        bail!(
            "installing from an IPSW needs reims-ctl signed with the com.apple.security.virtualization \
             entitlement: rerun scripts/install-ctl.sh (a plain cargo install leaves it out)"
        );
    }
    let exe = std::env::current_exe()?;
    let ipsw = if ipsw == "latest" {
        progress("asking Apple for the newest macOS this Mac supports");
        let out = Command::new(&exe).arg("__vz-latest").output()?;
        let url = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !out.status.success() || !url.starts_with("http") {
            bail!(
                "could not find the latest IPSW: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        download(&url, progress)?
    } else {
        PathBuf::from(ipsw)
    };
    if !ipsw.is_file() {
        bail!("{} does not exist", ipsw.display());
    }
    progress("loading the IPSW");
    let mut child = Command::new(&exe)
        .arg("__vz-install")
        .arg("--ipsw")
        .arg(&ipsw)
        .arg("--dir")
        .arg(dir)
        .args([
            "--disk-mib",
            &disk_mib.to_string(),
            "--cpus",
            &cpus.to_string(),
            "--memory-mib",
            &memory_mib.to_string(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stderr = child.stderr.take().unwrap();
    let err_thread = std::thread::spawn(move || {
        let mut s = String::new();
        use std::io::Read;
        let _ = std::io::BufReader::new(stderr).read_to_string(&mut s);
        s
    });
    let mut ecid = None;
    let mut last = -1i64;
    for line in std::io::BufReader::new(child.stdout.take().unwrap()).lines() {
        let line = line?;
        if let Some(f) = line.strip_prefix("progress ") {
            let pct = (f.parse::<f64>().unwrap_or(0.0) * 100.0) as i64;
            if pct != last {
                last = pct;
                progress(&format!("installing macOS: {pct}%"));
            }
        } else if let Some(e) = line.strip_prefix("ecid ") {
            ecid = e.trim().parse().ok();
        } else if let Some(m) = line.strip_prefix("note ") {
            progress(m);
        }
    }
    let status = child.wait()?;
    let err = err_thread.join().unwrap_or_default();
    match (status.success(), ecid) {
        (true, Some(e)) => Ok(e),
        _ => bail!("the install failed: {}", err.trim()),
    }
}

/// Downloads `url` into the data directory's `ipsw/`, resuming a partial download; returns the
/// file.
fn download(url: &str, progress: &mut dyn FnMut(&str)) -> Result<PathBuf> {
    let name = url
        .rsplit('/')
        .next()
        .filter(|n| n.ends_with(".ipsw"))
        .unwrap_or("latest.ipsw");
    let dir = crate::paths::data_dir().join("ipsw");
    std::fs::create_dir_all(&dir)?;
    let file = dir.join(name);
    if file.is_file() {
        progress(&format!(
            "using the IPSW downloaded before, {}",
            crate::paths::tilde(&file)
        ));
        return Ok(file);
    }
    let part = dir.join(format!("{name}.part"));
    let total: u64 = Command::new("curl")
        .args(["-sIL", url])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().to_string())
                })
                .next_back()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0);
    let mut child = Command::new("curl")
        .args(["-fsSL", "--retry", "3", "-C", "-", "-o"])
        .arg(&part)
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("run curl")?;
    loop {
        if let Some(st) = child.try_wait()? {
            if !st.success() {
                let mut e = String::new();
                use std::io::Read;
                let _ = child.stderr.take().map(|mut s| s.read_to_string(&mut e));
                bail!("the download failed: {}", e.trim());
            }
            break;
        }
        let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        if total > 0 {
            progress(&format!(
                "downloading {name}: {:.0}% of {:.1} GB",
                have as f64 * 100.0 / total as f64,
                total as f64 / 1e9
            ));
        } else {
            progress(&format!("downloading {name}: {:.1} GB", have as f64 / 1e9));
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    std::fs::rename(&part, &file)?;
    Ok(file)
}

#[cfg(target_os = "macos")]
mod imp {
    use anyhow::{Context, Result, bail};
    use block2::RcBlock;
    use objc2::AllocAnyThread;
    use objc2::rc::Retained;
    use objc2_foundation::{NSArray, NSDate, NSError, NSRunLoop, NSString, NSURL};
    use objc2_virtualization::*;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    fn url(p: &Path) -> Retained<NSURL> {
        NSURL::fileURLWithPath(&NSString::from_str(&p.to_string_lossy()))
    }

    fn describe(e: *mut NSError) -> String {
        if e.is_null() {
            return "unknown error".into();
        }
        // SAFETY: a non-null NSError the framework passed.
        unsafe { (*e).localizedDescription().to_string() }
    }

    fn spin_until<T>(slot: &Arc<Mutex<Option<T>>>) -> T {
        loop {
            if let Some(v) = slot.lock().unwrap().take() {
                return v;
            }
            NSRunLoop::mainRunLoop().runUntilDate(&NSDate::dateWithTimeIntervalSinceNow(0.1));
        }
    }

    /// Loads a restore image (from a file, or Apple's newest), as a raw retained pointer.
    fn restore_image(file: Option<&Path>) -> Result<Retained<VZMacOSRestoreImage>> {
        let slot: Arc<Mutex<Option<Result<usize, String>>>> = Arc::new(Mutex::new(None));
        let s = slot.clone();
        let block = RcBlock::new(move |img: *mut VZMacOSRestoreImage, err: *mut NSError| {
            let r = if img.is_null() {
                Err(describe(err))
            } else {
                // SAFETY: a live image the framework passed; retained here, released by the
                // Retained made from it below.
                Ok(unsafe { Retained::into_raw(Retained::retain(img).unwrap()) } as usize)
            };
            *s.lock().unwrap() = Some(r);
        });
        // SAFETY: the block lives until the call returns, and the framework copies it.
        unsafe {
            match file {
                Some(f) => VZMacOSRestoreImage::loadFileURL_completionHandler(&url(f), &block),
                None => VZMacOSRestoreImage::fetchLatestSupportedWithCompletionHandler(&block),
            }
        }
        let ptr = spin_until(&slot).map_err(anyhow::Error::msg)?;
        // SAFETY: retained in the block above.
        Ok(unsafe { Retained::from_raw(ptr as *mut VZMacOSRestoreImage) }.unwrap())
    }

    pub fn latest() -> Result<String> {
        let img = restore_image(None)?;
        // SAFETY: a loaded image.
        let u = unsafe { img.URL() };
        Ok(u.absoluteString()
            .map(|s| s.to_string())
            .unwrap_or_default())
    }

    pub fn install(
        ipsw: &Path,
        dir: &Path,
        disk_mib: u64,
        cpus: u32,
        memory_mib: u64,
    ) -> Result<u64> {
        let img = restore_image(Some(ipsw)).context("load the IPSW")?;
        // SAFETY: each call below is on the main thread, with live objects, in the order
        // Virtualization.framework's documentation gives.
        unsafe {
            let req = img.mostFeaturefulSupportedConfiguration().context(
                "this Mac cannot run the macOS in this IPSW (too old or too new for it)",
            )?;
            let hw = req.hardwareModel();
            if !hw.isSupported() {
                bail!("this Mac does not support the IPSW's hardware model");
            }
            let cpus = (cpus as usize).max(req.minimumSupportedCPUCount());
            let memory = (memory_mib * 1024 * 1024).max(req.minimumSupportedMemorySize());
            println!("note creating the auxiliary storage");
            let aux = VZMacAuxiliaryStorage::initCreatingStorageAtURL_hardwareModel_options_error(
                VZMacAuxiliaryStorage::alloc(),
                &url(&dir.join("aux.img")),
                &hw,
                VZMacAuxiliaryStorageInitializationOptions::AllowOverwrite,
            )
            .map_err(|e| anyhow::anyhow!("create aux.img: {}", e.localizedDescription()))?;
            let ident = VZMacMachineIdentifier::new();
            let ident_data = ident.dataRepresentation().to_vec();
            std::fs::write(dir.join("machine-identifier.bin"), &ident_data)?;
            std::fs::write(
                dir.join("hardware-model.bin"),
                hw.dataRepresentation().to_vec(),
            )?;
            let ecid = crate::create::ecid_from_identifier(&ident_data)?;

            let disk = dir.join("disk.img");
            std::fs::File::create(&disk)?.set_len(disk_mib * 1024 * 1024)?;

            let platform = VZMacPlatformConfiguration::new();
            platform.setHardwareModel(&hw);
            platform.setMachineIdentifier(&ident);
            platform.setAuxiliaryStorage(Some(&aux));
            let config = VZVirtualMachineConfiguration::new();
            config.setPlatform(&platform);
            config.setBootLoader(Some(&VZMacOSBootLoader::new()));
            config.setCPUCount(cpus);
            config.setMemorySize(memory);
            let attachment = VZDiskImageStorageDeviceAttachment::initWithURL_readOnly_error(
                VZDiskImageStorageDeviceAttachment::alloc(),
                &url(&disk),
                false,
            )
            .map_err(|e| anyhow::anyhow!("attach disk.img: {}", e.localizedDescription()))?;
            let blk = VZVirtioBlockDeviceConfiguration::initWithAttachment(
                VZVirtioBlockDeviceConfiguration::alloc(),
                &attachment,
            );
            config.setStorageDevices(&NSArray::from_retained_slice(&[Retained::into_super(blk)]));
            let display = VZMacGraphicsDisplayConfiguration::initWithWidthInPixels_heightInPixels_pixelsPerInch(
                VZMacGraphicsDisplayConfiguration::alloc(),
                1920,
                1200,
                80,
            );
            let gfx = VZMacGraphicsDeviceConfiguration::new();
            gfx.setDisplays(&NSArray::from_retained_slice(&[display]));
            config.setGraphicsDevices(&NSArray::from_retained_slice(&[Retained::into_super(gfx)]));
            config.validateWithError().map_err(|e| {
                anyhow::anyhow!("the VM's configuration: {}", e.localizedDescription())
            })?;

            let vm = VZVirtualMachine::initWithConfiguration(VZVirtualMachine::alloc(), &config);
            let installer = VZMacOSInstaller::initWithVirtualMachine_restoreImageURL(
                VZMacOSInstaller::alloc(),
                &vm,
                &url(ipsw),
            );
            let done: Arc<Mutex<Option<Result<(), String>>>> = Arc::new(Mutex::new(None));
            let d = done.clone();
            let block = RcBlock::new(move |err: *mut NSError| {
                *d.lock().unwrap() = Some(if err.is_null() {
                    Ok(())
                } else {
                    Err(describe(err))
                });
            });
            println!("note installing macOS");
            installer.installWithCompletionHandler(&block);
            let progress = installer.progress();
            let mut last = -1.0;
            let result = loop {
                if let Some(r) = done.lock().unwrap().take() {
                    break r;
                }
                let f = progress.fractionCompleted();
                if (f - last).abs() >= 0.001 {
                    last = f;
                    println!("progress {f:.4}");
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                }
                NSRunLoop::mainRunLoop().runUntilDate(&NSDate::dateWithTimeIntervalSinceNow(0.25));
            };
            result.map_err(anyhow::Error::msg)?;
            Ok(ecid)
        }
    }
}

/// `reims-ctl __vz-install`.
pub fn install_main(
    ipsw: &Path,
    dir: &Path,
    disk_mib: u64,
    cpus: u32,
    memory_mib: u64,
) -> Result<i32> {
    #[cfg(target_os = "macos")]
    {
        let ecid = imp::install(ipsw, dir, disk_mib, cpus, memory_mib)?;
        println!("ecid {ecid}");
        Ok(0)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (ipsw, dir, disk_mib, cpus, memory_mib);
        bail!("an IPSW installs only on macOS")
    }
}

/// `reims-ctl __vz-latest`: the URL of the newest IPSW this Mac supports.
pub fn latest_main() -> Result<i32> {
    #[cfg(target_os = "macos")]
    {
        println!("{}", imp::latest()?);
        Ok(0)
    }
    #[cfg(not(target_os = "macos"))]
    bail!("IPSWs are macOS's")
}
