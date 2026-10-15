using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace ReimsManager;

internal static class GuestClipboard
{
    internal const int MaxBytes = 1024 * 1024;

    internal static async Task<string> Transfer(string vmDirectory, string? text, CancellationToken cancellation = default)
    {
        if (text is not null && Encoding.UTF8.GetByteCount(text) > MaxBytes)
            throw new InvalidOperationException("共有できるテキストは 1 MiB までです。");
        string state = Path.Combine(vmDirectory, "state");
        using var config = JsonDocument.Parse(File.ReadAllText(Path.Combine(state, "config.json")));
        using var paired = JsonDocument.Parse(File.ReadAllText(Path.Combine(state, "guest.json")));
        using var runtime = JsonDocument.Parse(File.ReadAllText(Path.Combine(state, "runtime.json")));
        var r = runtime.RootElement;
        using var vm = Process.GetProcessById(r.GetProperty("pid").GetInt32());
        if (vm.StartTime.ToUniversalTime().Ticks != r.GetProperty("start_ticks").GetInt64() ||
            !string.Equals(vm.MainModule?.FileName, r.GetProperty("executable").GetString(), StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("選択した VM のプロセスを確認できません。");
        var c = config.RootElement;
        string host = "127.0.0.1", port = c.GetProperty("ssh_port").ToString();
        if (c.TryGetProperty("network", out var network) && network.GetProperty("mode").GetString() == "bridge") {
            host = network.GetProperty("guest_host").GetString() ?? "";
            port = "22";
        }
        string user = paired.RootElement.GetProperty("user").GetString() ?? "";
        if (!Regex.IsMatch(user, "^[a-z_][a-z0-9_-]*$") || !Regex.IsMatch(host, "^[A-Za-z0-9][A-Za-z0-9.:-]*$"))
            throw new InvalidOperationException("macOS 接続の設定を確認してください。");
        var info = new ProcessStartInfo(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Windows), "System32/OpenSSH/ssh.exe")) {
            UseShellExecute = false, CreateNoWindow = true, RedirectStandardInput = true,
            RedirectStandardOutput = true, RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8
        };
        foreach (string arg in new[] { "-F", "NUL", "-T", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes", "-o", "KexAlgorithms=curve25519-sha256",
            "-o", "UserKnownHostsFile=" + Path.Combine(state, "known_hosts"), "-p", port,
            "-i", Path.Combine(state, "ssh/id_ed25519") }) info.ArgumentList.Add(arg);
        if (paired.RootElement.TryGetProperty("host_key_alias", out var alias)) {
            info.ArgumentList.Add("-o"); info.ArgumentList.Add("HostKeyAlias=" + alias.GetString());
        }
        info.ArgumentList.Add(user + "@" + host);
        info.ArgumentList.Add(text is null ? "LC_CTYPE=en_US.UTF-8 /usr/bin/pbpaste" : "/usr/bin/base64 -D | LC_CTYPE=en_US.UTF-8 /usr/bin/pbcopy");
        using var process = Process.Start(info)!;
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
        timeout.CancelAfter(TimeSpan.FromSeconds(10));
        using var kill = timeout.Token.Register(() => { try { if (!process.HasExited) process.Kill(); } catch (InvalidOperationException) { } catch (System.ComponentModel.Win32Exception) { } });
        var error = process.StandardError.ReadToEndAsync(timeout.Token);
        // Clipboard contents travel only through stdin/stdout, never argv or logs.
        var output = ReadBounded(process.StandardOutput, timeout.Token);
        if (text is not null) await process.StandardInput.WriteAsync(Convert.ToBase64String(Encoding.UTF8.GetBytes(text)).AsMemory(), timeout.Token);
        process.StandardInput.Close();
        try {
            string result = await output;
            await process.WaitForExitAsync(timeout.Token);
            await error;
            if (process.ExitCode != 0) throw new InvalidOperationException("クリップボード接続に失敗しました。VM の起動と SSH ペアリングを確認してください。");
            return result;
        } finally { if (!process.HasExited) { process.Kill(); await process.WaitForExitAsync(); } }
    }

    private static async Task<string> ReadBounded(StreamReader reader, CancellationToken token)
    {
        var result = new StringBuilder();
        char[] buffer = new char[4096];
        int bytes = 0, count;
        while ((count = await reader.ReadAsync(buffer.AsMemory(), token)) != 0) {
            bytes += Encoding.UTF8.GetByteCount(buffer, 0, count);
            if (bytes > MaxBytes) throw new InvalidOperationException("共有できるテキストは 1 MiB までです。");
            result.Append(buffer, 0, count);
        }
        return result.ToString();
    }
}
