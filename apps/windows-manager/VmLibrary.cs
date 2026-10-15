using System.IO;
using System.Text.Json;

namespace ReimsManager;

internal sealed record VmEntry(string Name, string File, string Directory)
{
    public override string ToString() => Name;
}

internal static class VmLibrary
{
    internal static VmEntry Open(string file)
    {
        file = Path.GetFullPath(file);
        using var json = JsonDocument.Parse(System.IO.File.ReadAllText(file));
        var value = json.RootElement;
        if (value.GetProperty("schema").GetInt32() != 1) throw new InvalidDataException("この VM ファイルの形式には対応していません。");
        string name = value.GetProperty("name").GetString() ?? "";
        string directory = Path.GetFullPath(value.GetProperty("directory").GetString()!, Path.GetDirectoryName(file)!);
        string package = Path.GetDirectoryName(file)!;
        if (!string.Equals(directory,package,StringComparison.OrdinalIgnoreCase) &&
            !directory.StartsWith(package.TrimEnd(Path.DirectorySeparatorChar)+Path.DirectorySeparatorChar,StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("VM のデータは VM ファイルと同じフォルダー内に置いてください。");
        if (string.IsNullOrWhiteSpace(name) || directory.TrimEnd(Path.DirectorySeparatorChar) == Path.GetPathRoot(directory)?.TrimEnd(Path.DirectorySeparatorChar))
            throw new InvalidDataException("VM の名前または保存先が不正です。");
        return new(name, file, directory);
    }

    internal static VmEntry Create(string file, string directory)
    {
        file = Path.GetFullPath(file);
        if (System.IO.File.Exists(file)) throw new IOException("VM ファイルは既に存在します。");
        System.IO.Directory.CreateDirectory(Path.GetDirectoryName(file)!);
        string name = Path.GetFileNameWithoutExtension(file);
        string relative = Path.GetRelativePath(Path.GetDirectoryName(file)!, directory);
        using (var stream = new FileStream(file, FileMode.CreateNew, FileAccess.Write))
            JsonSerializer.Serialize(stream, new { schema = 1, name, directory = relative });
        return Open(file);
    }
}
