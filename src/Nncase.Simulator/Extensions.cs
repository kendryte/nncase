using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace Nncase.Simulator;

public static class SimulatorExtension
{
    public static string ToFile<T>(this Tensor<T> tensor, string path)
      where T : unmanaged, System.IEquatable<T>
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

