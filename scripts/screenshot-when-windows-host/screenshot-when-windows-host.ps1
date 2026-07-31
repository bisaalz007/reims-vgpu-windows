# screenshot-when-windows-host.ps1 — the capture half of the Windows screenshot
# script. Invoked by the .sh wrapper; not meant to be run directly.
#
# Uses PrintWindow with PW_RENDERFULLCONTENT rather than CopyFromScreen. The
# screen-copy approach captures whatever pixels happen to be at those
# coordinates, so any window that overlaps the VM — a chat client, an editor —
# lands in the "screenshot" and is indistinguishable from guest output. That is
# the one failure mode a visual regression gate cannot tolerate, and it happened
# on the first capture taken on this rail. PrintWindow asks the window to draw
# itself, so an occluded window still yields its own contents.
param(
    [string]$Out = "",
    # Named TargetPid, not Pid: $PID is a read-only automatic variable in
    # PowerShell, and binding a parameter to it fails at invocation.
    [int]$TargetPid = 0,
    [int]$MaxWidth = 1280,
    [int]$MaxHeight = 720
)
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class ReimsCap {
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr dc, uint flags);
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
}
"@

# Window rects are physical pixels. Without this the capture is offset and
# clipped on any display with scaling, which is the default on modern Windows.
[void][ReimsCap]::SetProcessDPIAware()

$procs = @(Get-Process -Name "qemu-system-*" -ErrorAction SilentlyContinue |
           Where-Object { $_.MainWindowHandle -ne [IntPtr]::Zero })
if ($TargetPid -ne 0) { $procs = @($procs | Where-Object { $_.Id -eq $TargetPid }) }

if ($procs.Count -eq 0) {
    Write-Error "no qemu-system-* process with a window (is the VM running, and is -display none off?)"
    exit 1
}
if ($procs.Count -gt 1 -and $TargetPid -eq 0) {
    $ids = ($procs | ForEach-Object { $_.Id }) -join ", "
    Write-Error "several QEMU windows are open (pids: $ids); pass --pid to choose one"
    exit 1
}
$h = $procs[0].MainWindowHandle

if ([ReimsCap]::IsIconic($h)) {
    Write-Error "the QEMU window is minimised; PrintWindow cannot capture it"
    exit 1
}

$r = New-Object ReimsCap+RECT
[void][ReimsCap]::GetWindowRect($h, [ref]$r)
$w = $r.R - $r.L
$hh = $r.B - $r.T
if ($w -le 0 -or $hh -le 0) { Write-Error "window has no area"; exit 1 }

$bmp = New-Object System.Drawing.Bitmap $w, $hh
$g = [System.Drawing.Graphics]::FromImage($bmp)
$dc = $g.GetHdc()
# 0x00000002 = PW_RENDERFULLCONTENT, needed for windows that draw through a
# compositor surface; GTK's does.
$ok = [ReimsCap]::PrintWindow($h, $dc, 2)
$g.ReleaseHdc($dc)
$g.Dispose()
if (-not $ok) { $bmp.Dispose(); Write-Error "PrintWindow failed"; exit 1 }

# Downscale to the same budget the other two screenshot scripts use, so an agent
# reading one of these pays the same token cost whichever host produced it.
$scale = [Math]::Min(1.0, [Math]::Min($MaxWidth / [double]$w, $MaxHeight / [double]$hh))
if ($scale -lt 1.0) {
    $nw = [int]($w * $scale); $nh = [int]($hh * $scale)
    $small = New-Object System.Drawing.Bitmap $nw, $nh
    $sg = [System.Drawing.Graphics]::FromImage($small)
    $sg.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $sg.DrawImage($bmp, 0, 0, $nw, $nh)
    $sg.Dispose(); $bmp.Dispose()
    $bmp = $small
}

$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$size = "$($bmp.Width)x$($bmp.Height)"
$bmp.Dispose()
Write-Output $Out
[Console]::Error.WriteLine("screenshot-when-windows-host: $Out ($size, pid $($procs[0].Id))")
