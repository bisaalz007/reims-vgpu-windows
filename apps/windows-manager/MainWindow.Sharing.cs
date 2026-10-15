using System.Text.Json;
using System.Windows;
using System.Windows.Threading;

namespace ReimsManager;

public partial class MainWindow
{
    private readonly DispatcherTimer clipboardTimer = new() { Interval = TimeSpan.FromSeconds(2) };
    private CancellationTokenSource clipboardCancel = new();
    private bool clipboardBusy;
    private string? lastClipboard;

    private void ResetClipboard() { clipboardCancel.Cancel(); clipboardCancel.Dispose(); clipboardCancel = new(); lastClipboard = null; }
    private void ClipboardChanged(object sender, RoutedEventArgs e)
    {
        if (ClipboardStatus is null || selfTest) return;
        if (ClipboardSharing.IsChecked == true) {
            lastClipboard = null; clipboardTimer.Start(); ClipboardStatus.Text = "テキスト共有を開始します。";
        } else { clipboardTimer.Stop(); ResetClipboard(); ClipboardStatus.Text = "自動共有はオフです。"; }
    }

    private async Task SyncClipboard()
    {
        if (clipboardBusy || busy || ClipboardSharing.IsChecked != true || latest.ValueKind != JsonValueKind.Object || latest.GetProperty("state").GetString() != "running") return;
        clipboardBusy = true;
        string directory = VmDirectory;
        var token = clipboardCancel.Token;
        try {
            string? host = Clipboard.ContainsText() ? Clipboard.GetText() : null;
            if (host is not null && host != lastClipboard) {
                await GuestClipboard.Transfer(directory, host, token);
                lastClipboard = host;
            } else {
                string guest = await GuestClipboard.Transfer(directory, null, token);
                token.ThrowIfCancellationRequested();
                string? current = Clipboard.ContainsText() ? Clipboard.GetText() : null;
                // A new local copy made during SSH always wins over an older guest read.
                if (guest != lastClipboard && current == host && guest.Length > 0) Clipboard.SetText(guest);
                if (current == host) lastClipboard = guest;
            }
            ClipboardStatus.Text = "テキストを共有しています。";
        } catch (OperationCanceledException) { }
        catch (Exception error) { ClipboardStatus.Text = error.Message; }
        finally { clipboardBusy = false; }
    }

    private async void CopyToGuest(object sender, RoutedEventArgs e)
    {
        if (clipboardBusy) return; clipboardBusy = true;
        try { if (Clipboard.ContainsText()) { await GuestClipboard.Transfer(VmDirectory, Clipboard.GetText(), clipboardCancel.Token); ClipboardStatus.Text = "macOS にコピーしました。"; } }
        catch (Exception error) { ClipboardStatus.Text = error.Message; }
        finally { clipboardBusy = false; }
    }
    private async void CopyFromGuest(object sender, RoutedEventArgs e)
    {
        if (clipboardBusy) return; clipboardBusy = true;
        var token = clipboardCancel.Token;
        try { string text = await GuestClipboard.Transfer(VmDirectory, null, token); token.ThrowIfCancellationRequested(); if (text.Length > 0) Clipboard.SetText(text); ClipboardStatus.Text = "Windows にコピーしました。"; }
        catch (OperationCanceledException) { }
        catch (Exception error) { ClipboardStatus.Text = error.Message; }
        finally { clipboardBusy = false; }
    }

    private async Task ReadIdentity()
    {
        try {
            string requestedVm = VmDirectory;
            using var data = JsonDocument.Parse(await Execute("GetIdentity", []));
            if (requestedVm != VmDirectory) return;
            var i = data.RootElement;
            IdentityModel.Text = i.GetProperty("model").GetString(); IdentitySerial.Text = i.GetProperty("serial").GetString();
            IdentityMlb.Text = i.GetProperty("mlb").GetString(); IdentityUuid.Text = i.GetProperty("uuid").GetString(); IdentityRom.Text = i.GetProperty("rom").GetString();
        } catch (Exception error) { if (latest.ValueKind == JsonValueKind.Object && latest.GetProperty("prepared").GetBoolean()) State.Text = error.Message; }
    }
    private async void LoadIdentityClick(object sender, RoutedEventArgs e) => await ReadIdentity();
    private async void SaveIdentity(object sender, RoutedEventArgs e)
    {
        string data = JsonSerializer.Serialize(new { model = IdentityModel.Text, serial = IdentitySerial.Text, mlb = IdentityMlb.Text, uuid = IdentityUuid.Text, rom = IdentityRom.Text });
        if (Confirm("この VM の識別情報を変更します。Apple Account の再認証が必要になることがあります。"))
            if (await Operate("SetIdentity", input: data)) await ReadIdentity();
    }

    private sealed record UsbEntry(string Id, string Label);
    private async void InstallUsbDk(object sender, RoutedEventArgs e) => await Operate("InstallUsbDk", elevate: true);
    private async void RefreshUsb(object sender, RoutedEventArgs e)
    {
        try {
            string requestedVm = VmDirectory;
            using var list = JsonDocument.Parse(await Execute("ListUsb", []));
            if (requestedVm != VmDirectory) return;
            UsbDevices.ItemsSource = list.RootElement.EnumerateArray().Select(x => new UsbEntry(x.GetProperty("id").GetString()!, x.GetProperty("label").GetString()!)).ToArray();
            UsbHint.Text = "登録した機器は同じ USB ポートへの抜き差し後も自動接続します。別のポートでは選び直してください。新しい機器の検索には VM の起動が必要です。";
        } catch (Exception error) { UsbHint.Text = error.Message; }
    }
    private async void AttachUsb(object sender, RoutedEventArgs e) { if (UsbDevices.SelectedItem is UsbEntry u && await Operate("AttachUsb", ["-UsbId", u.Id])) RefreshUsb(sender, e); }
    private async void DetachUsb(object sender, RoutedEventArgs e) { if (UsbDevices.SelectedItem is UsbEntry u && await Operate("DetachUsb", ["-UsbId", u.Id])) RefreshUsb(sender, e); }
}
