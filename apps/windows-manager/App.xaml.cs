using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Windows;

namespace ReimsManager;

public partial class App : Application
{
    protected override async void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        try {
            if (e.Args.Length >= 2 && e.Args[0] == "--echo-args") {
                File.WriteAllText(e.Args[1], JsonSerializer.Serialize(e.Args.Skip(2))); Shutdown(0); return;
            }
            if (e.Args.Length==2 && e.Args[0]=="--keyboard-probe") {
                var probe=new KeyboardProbe(Path.GetFullPath(e.Args[1])); MainWindow=probe; probe.Show(); return;
            }
            if (e.Args.Length==3 && e.Args[0]=="--clipboard-check") {
                string folder=Path.GetFullPath(e.Args[2]); Directory.CreateDirectory(folder);
                const string marker="Reims 日本語 clipboard\n\"'$(literal); & < > _\r\n";
                string previous=await GuestClipboard.Transfer(e.Args[1],null);
                try {
                    await GuestClipboard.Transfer(e.Args[1],marker);
                    if (await GuestClipboard.Transfer(e.Args[1],null)!=marker) throw new Exception("Guest clipboard roundtrip failed");
                    File.WriteAllText(Path.Combine(folder,"clipboard-result.json"),"{\"unicode_roundtrip\":true,\"shell_literals\":true}");
                } finally { if (await GuestClipboard.Transfer(e.Args[1],null)==marker) await GuestClipboard.Transfer(e.Args[1],previous); }
                Shutdown(0); return;
            }
            var window = new MainWindow(e.Args.Length==2 && e.Args[0]=="--self-test"); MainWindow = window;
            if (e.Args.Length == 2 && e.Args[0] == "--root") window.Root.Text = Path.GetFullPath(e.Args[1]);
            if (e.Args.Length == 2 && e.Args[0] == "--self-test") {
                string folder = Path.GetFullPath(e.Args[1]); Directory.CreateDirectory(folder);
                string[] values = ["", "日本語 path", "quote\"value", "trailing\\", "$(do-nothing)&literal", "comma,path"];
                string output = Path.Combine(folder, "argv.json");
                var info = new ProcessStartInfo(Environment.ProcessPath!) { UseShellExecute=false, CreateNoWindow=true };
                foreach (string arg in new[] {"--echo-args",output}.Concat(values)) info.ArgumentList.Add(arg);
                using var child = Process.Start(info)!; await child.WaitForExitAsync();
                if (child.ExitCode != 0 || !values.SequenceEqual(JsonSerializer.Deserialize<string[]>(File.ReadAllText(output))!)) throw new Exception("Argument roundtrip failed");
                window.Opacity = 0; window.ShowInTaskbar = false; window.Show();
                await window.RenderChecks(folder);
                string fixture=Path.Combine(folder,"fixture.reimsvm");
                if (File.Exists(fixture)) File.Delete(fixture);
                var entry=VmLibrary.Create(fixture,Path.Combine(folder,"fixture.data"));
                if (VmLibrary.Open(fixture)!=entry) throw new Exception("VM library roundtrip failed");
                File.WriteAllText(Path.Combine(folder,"result.json"),JsonSerializer.Serialize(new {fluent_theme=true,argument_roundtrip=true,vm_library_roundtrip=true,pages_rendered=window.Tabs.Items.Count+window.Devices.Items.Count-1}));
                window.Close(); Shutdown(0); return;
            }
            window.Show();
        } catch (Exception error) {
            if (e.Args.Length > 0) {
                File.WriteAllText(Path.Combine(Path.GetFullPath(e.Args[^1]), "error.txt"), error.ToString()); Shutdown(1);
            } else { MessageBox.Show(error.Message,"Reims Manager"); Shutdown(1); }
        }
    }
}
