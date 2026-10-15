using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;

namespace ReimsManager;

public sealed class VmConsoleHost : HwndHost
{
    private nint child, originalStyle;
    private RectNative originalBounds;
    public bool Attached => child != 0 && IsWindow(child);

    internal bool Attach(int pid, long startTicks, string executable)
    {
        if (Handle == 0) return false;
        if (Attached) {
            GetWindowThreadProcessId(child, out uint owner);
            if (owner == pid) return true;
            Detach();
        }
        using var process = Process.GetProcessById(pid);
        if (process.StartTime.ToUniversalTime().Ticks != startTicks ||
            !string.Equals(process.MainModule?.FileName, executable, StringComparison.OrdinalIgnoreCase)) return false;
        nint candidate = 0;
        EnumWindows((window, _) => {
            GetWindowThreadProcessId(window, out uint owner);
            if (owner == pid && IsWindowVisible(window)) { candidate = window; return false; }
            return true;
        }, 0);
        if (candidate == 0) return false;
        originalStyle = GetWindowLongPtr(candidate, -16);
        GetWindowRect(candidate, out originalBounds);
        Marshal.SetLastPInvokeError(0);
        if (SetParent(candidate, Handle) == 0 && Marshal.GetLastPInvokeError() != 0)
            throw new InvalidOperationException("VM 画面を接続できませんでした。別ウィンドウ表示を使用してください。");
        child = candidate;
        SetWindowLongPtr(child, -16, (nint)(((long)originalStyle & ~0x80CF0000L) | 0x40000000L));
        ResizeGuest();
        return true;
    }

    internal void Detach()
    {
        if (Attached) {
            SetParent(child, 0);
            SetWindowLongPtr(child, -16, originalStyle);
            SetWindowPos(child, 0, originalBounds.Left, originalBounds.Top,
                originalBounds.Right-originalBounds.Left, originalBounds.Bottom-originalBounds.Top, 0x0034);
            ShowWindow(child, 5);
        }
        child = 0;
    }

    private void ResizeGuest()
    {
        if (!Attached || Handle == 0) return;
        GetClientRect(Handle, out var bounds);
        SetWindowPos(child, 0, 0, 0, bounds.Right, bounds.Bottom, 0x0034);
    }

    protected override HandleRef BuildWindowCore(HandleRef parent) => new(this,
        CreateWindowEx(0, "static", "Reims VM console", 0x56000000, 0, 0, 1, 1, parent.Handle, 0, 0, 0));
    protected override void DestroyWindowCore(HandleRef window) { Detach(); DestroyWindow(window.Handle); }
    protected override void OnWindowPositionChanged(Rect rectangle) { base.OnWindowPositionChanged(rectangle); ResizeGuest(); }

    [StructLayout(LayoutKind.Sequential)] private struct RectNative { public int Left, Top, Right, Bottom; }
    private delegate bool EnumCallback(nint window, nint param);
    [DllImport("user32.dll")] private static extern bool EnumWindows(EnumCallback callback, nint param);
    [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(nint window, out uint process);
    [DllImport("user32.dll")] private static extern bool IsWindow(nint window);
    [DllImport("user32.dll")] private static extern bool IsWindowVisible(nint window);
    [DllImport("user32.dll", SetLastError=true)] private static extern nint SetParent(nint child, nint parent);
    [DllImport("user32.dll", EntryPoint="GetWindowLongPtrW")] private static extern nint GetWindowLongPtr(nint window, int index);
    [DllImport("user32.dll", EntryPoint="SetWindowLongPtrW")] private static extern nint SetWindowLongPtr(nint window, int index, nint value);
    [DllImport("user32.dll")] private static extern bool GetWindowRect(nint window, out RectNative bounds);
    [DllImport("user32.dll")] private static extern bool GetClientRect(nint window, out RectNative bounds);
    [DllImport("user32.dll")] private static extern bool SetWindowPos(nint window, nint after, int x, int y, int width, int height, uint flags);
    [DllImport("user32.dll")] private static extern bool ShowWindow(nint window, int command);
    [DllImport("user32.dll")] private static extern bool DestroyWindow(nint window);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] private static extern nint CreateWindowEx(int extended, string cls, string name, int style, int x, int y, int width, int height, nint parent, nint menu, nint instance, nint param);
}
