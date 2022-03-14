
namespace Nncase.Simulator;

public static class SimulatorExtension
{
    public static string ToFile(this Tensor tensor, string path)
    {
        if (File.Exists(path))
            File.Delete(path);
        if (Path.GetDirectoryName(path) is var dir && path is not null && path != string.Empty && !Directory.Exists(dir))
            Directory.CreateDirectory(dir!);

        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        writer.Write(tensor.BytesBuffer);
        return path;
    }
}

