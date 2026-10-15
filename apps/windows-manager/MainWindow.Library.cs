using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using Microsoft.Win32;

namespace ReimsManager;

public partial class MainWindow
{
    private readonly List<VmEntry> machines = [];
    private bool consoleDetached;
    private string VmDirectory => (Library?.SelectedItem as VmEntry)?.Directory ?? Path.Combine(Root.Text, "vm");
    private string LibraryFile => Path.Combine(Root.Text, ".work/manager-library.json");

    private void LoadLibrary()
    {
        machines.Clear();
        if (File.Exists(LibraryFile)) {
            foreach (string path in JsonSerializer.Deserialize<string[]>(File.ReadAllText(LibraryFile)) ?? []) {
                try { machines.Add(VmLibrary.Open(path)); }
                catch (Exception error) { Log.AppendText($"VM を読み込めません: {path}\n{error.Message}\n"); }
            }
        }
        string legacy = Path.Combine(Root.Text, "vm");
        if (File.Exists(Path.Combine(legacy, "state/config.json")) && !machines.Any(x => string.Equals(x.Directory, legacy, StringComparison.OrdinalIgnoreCase))) {
            string launcher = Path.Combine(legacy, "macOS Tahoe.reimsvm");
            machines.Add(File.Exists(launcher) ? VmLibrary.Open(launcher) : VmLibrary.Create(launcher, legacy));
        }
        Library.ItemsSource = machines.ToArray();
        if (machines.Count > 0) Library.SelectedIndex = 0;
    }

    private void AddVm(VmEntry entry)
    {
        if (!machines.Any(x => string.Equals(x.File, entry.File, StringComparison.OrdinalIgnoreCase))) machines.Add(entry);
        Directory.CreateDirectory(Path.GetDirectoryName(LibraryFile)!);
        File.WriteAllText(LibraryFile + ".new", JsonSerializer.Serialize(machines.Select(x => x.File)));
        File.Move(LibraryFile + ".new", LibraryFile, true);
        Library.ItemsSource = machines.ToArray(); Library.SelectedItem = machines.First(x => x.File == entry.File);
    }

    private async void VmChanged(object sender, SelectionChangedEventArgs e)
    {
        if (Tabs is null || busy || Library.SelectedItem is not VmEntry entry) return;
        ConsoleView.Detach(); consoleDetached = false;
        ClipboardSharing.IsChecked = false; ResetClipboard();
        loadedHardware = false; latest = default;
        VmTitle.Text = entry.Name; VmLocation.Text = entry.Directory;
        IdentityModel.Clear(); IdentitySerial.Clear(); IdentityMlb.Clear(); IdentityRom.Clear(); IdentityUuid.Clear();
        UsbDevices.ItemsSource = null;
        if (!selfTest) { await RefreshState(); await ReadIdentity(); }
    }

    private void OpenVm(object sender, RoutedEventArgs e)
    {
        if (busy) return;
        var dialog = new OpenFileDialog { Filter = "Reims 仮想マシン|*.reimsvm" };
        if (dialog.ShowDialog(this) != true) return;
        try { AddVm(VmLibrary.Open(dialog.FileName)); } catch (Exception error) { State.Text = error.Message; }
    }

    private VmEntry? CreateVm()
    {
        var dialog = new SaveFileDialog { Filter = "Reims 仮想マシン|*.reimsvm", FileName = "macOS.reimsvm", Title = "VM の名前と保存先を選択", OverwritePrompt = true };
        if (dialog.ShowDialog(this) != true) return null;
        string directory = Path.Combine(Path.GetDirectoryName(dialog.FileName)!, Path.GetFileNameWithoutExtension(dialog.FileName) + ".data");
        try {
            if (Directory.Exists(directory) && Directory.EnumerateFileSystemEntries(directory).Any()) throw new IOException("この保存先には既にデータがあります。別の VM 名を選択してください。");
            var entry = VmLibrary.Create(dialog.FileName, directory); AddVm(entry); return entry;
        }
        catch (Exception error) { State.Text = error.Message; return null; }
    }

    private void NewVm(object sender, RoutedEventArgs e) { if (!busy && CreateVm() is not null) Tabs.SelectedIndex = 5; }
    private async void ImportVm(object sender, RoutedEventArgs e)
    {
        if (busy) return;
        var disk = new OpenFileDialog { Filter = "仮想ディスク|*.vmdk;*.qcow2;*.img;*.raw", Title = "停止済みの VM のディスクを選択" };
        if (disk.ShowDialog(this) != true || CreateVm() is null) return;
        if (await Operate("ImportDisk", ["-DiskPath", disk.FileName])) { loadedHardware = false; await RefreshState(); await ReadIdentity(); Tabs.SelectedIndex = 0; }
    }

    private void RevealVm(object sender, RoutedEventArgs e)
    {
        string folder = Directory.Exists(VmDirectory) ? VmDirectory : Path.GetDirectoryName((Library.SelectedItem as VmEntry)?.File ?? Root.Text)!;
        if (Directory.Exists(folder)) Process.Start(new ProcessStartInfo(folder) { UseShellExecute = true });
    }
    private void ShowPowerMenu(object sender, RoutedEventArgs e) { var menu = ((Button)sender).ContextMenu; menu.PlacementTarget = (Button)sender; menu.IsOpen = true; }
    private async void MenuAction(object sender, RoutedEventArgs e) => await Operate(((MenuItem)sender).Tag.ToString()!);
    private void OpenSoundSettings(object sender, RoutedEventArgs e) => Process.Start(new ProcessStartInfo("ms-settings:sound") { UseShellExecute = true });
    private void PopOutConsole(object sender, RoutedEventArgs e) { consoleDetached = true; ConsoleView.Detach(); ConsoleView.Visibility = Visibility.Hidden; ConsolePlaceholder.Visibility = Visibility.Visible; }
    private void DockConsole(object sender, RoutedEventArgs e) { consoleDetached = false; Tabs.SelectedIndex = 0; UpdateConsole(); }
    private void TabChanged(object sender, SelectionChangedEventArgs e) { if (e.Source == Tabs && ConsoleView is not null) UpdateConsole(); }

    private void UpdateConsole()
    {
        if (selfTest || consoleDetached || Tabs.SelectedIndex != 0 || latest.ValueKind != JsonValueKind.Object) return;
        if (latest.GetProperty("state").GetString() == "shutdown") {
            ConsoleView.Visibility = Visibility.Hidden; ConsolePlaceholder.Visibility = Visibility.Visible; return;
        }
        if (latest.TryGetProperty("runtime", out var runtime) && runtime.ValueKind == JsonValueKind.Object) {
            try {
                ConsoleView.Visibility = Visibility.Visible;
                if (ConsoleView.Attach(runtime.GetProperty("pid").GetInt32(), runtime.GetProperty("start_ticks").GetInt64(), runtime.GetProperty("executable").GetString()!)) {
                    ConsolePlaceholder.Visibility = Visibility.Collapsed; return;
                }
            } catch (Exception error) { Log.AppendText(error.Message + "\n"); consoleDetached = true; }
        }
        ConsoleView.Visibility = Visibility.Hidden; ConsolePlaceholder.Visibility = Visibility.Visible;
    }
}
