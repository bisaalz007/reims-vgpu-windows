using System.IO;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Interop;

namespace ReimsManager;

// Foreground-only metadata probe. It records special-key codes, never text.
internal sealed class KeyboardProbe : Window
{
    [StructLayout(LayoutKind.Sequential)] private struct RawDevice { public ushort page, usage; public uint flags; public nint target; }
    [DllImport("user32.dll", SetLastError=true)] private static extern bool RegisterRawInputDevices(RawDevice[] devices,uint count,uint size);
    [DllImport("user32.dll", SetLastError=true)] private static extern uint GetRawInputData(nint input,uint command,nint data,ref uint size,uint headerSize);
    private readonly TextBox output = new() { IsReadOnly=true, AcceptsReturn=true, VerticalScrollBarVisibility=ScrollBarVisibility.Auto, Height=180 };
    private readonly string path;
    internal KeyboardProbe(string folder)
    {
        Directory.CreateDirectory(folder); path=Path.Combine(folder,"special-keys.log");
        Title="英数・かなの入力確認"; Width=620; Height=350; WindowStartupLocation=WindowStartupLocation.CenterScreen;
        var panel=new StackPanel { Margin=new Thickness(24) };
        panel.Children.Add(new TextBlock { Text="このウィンドウをクリックしてから、英数・かなを一回ずつ押してください。", TextWrapping=TextWrapping.Wrap, FontSize=18, Margin=new Thickness(0,0,0,16) });
        panel.Children.Add(output); Content=panel;
        SourceInitialized += (_,_) => {
            var source=(HwndSource)PresentationSource.FromVisual(this)!; source.AddHook(Message);
            RawDevice[] devices=[new() {page=1,usage=6,target=source.Handle}];
            if (!RegisterRawInputDevices(devices,1,(uint)Marshal.SizeOf<RawDevice>())) throw new System.ComponentModel.Win32Exception();
        };
    }
    private static bool Special(int scan,int key) => scan>=0x70 || key>=0xf0 || key is 0x14 or 0x15 or 0x19 or 0x1c or 0x1d or 0x5b or 0x5c;
    private void Record(string line) { File.AppendAllText(path,line+Environment.NewLine); output.AppendText(line+"\n"); output.ScrollToEnd(); }
    private nint Message(nint window,int message,nint wparam,nint lparam,ref bool handled)
    {
        if (message==0x100 || message==0x101 || message==0x104 || message==0x105) {
            int key=(int)wparam, scan=(int)(((long)lparam>>16)&0xff);
            if (Special(scan,key)) Record($"message=0x{message:x} scan=0x{scan:x} vk=0x{key:x}");
        }
        if (message==0xff) {
            uint header=(uint)(IntPtr.Size==8?24:16), size=0;
            if (GetRawInputData(lparam,0x10000003,0,ref size,header)!=0 || size<header+16) return 0;
            nint buffer=Marshal.AllocHGlobal((int)size);
            try {
                if (GetRawInputData(lparam,0x10000003,buffer,ref size,header)==uint.MaxValue || Marshal.ReadInt32(buffer)!=1) return 0;
                nint keyboard=buffer+(int)header;
                int scan=(ushort)Marshal.ReadInt16(keyboard), flags=(ushort)Marshal.ReadInt16(keyboard,2), key=(ushort)Marshal.ReadInt16(keyboard,6);
                if (Special(scan,key)) Record($"raw scan=0x{scan:x} vk=0x{key:x} flags=0x{flags:x}");
            } finally { Marshal.FreeHGlobal(buffer); }
        }
        return 0;
    }
}
