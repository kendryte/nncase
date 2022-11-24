using System.Text;
using Nncase.IR;

namespace Nncase.Utilities;

public class DumpManager
{
    public static bool OpenDump { get; private set; } = false;

    public static bool Append = false;

    public static int Count = 1;

    public static string Dir;

    public string CountStr => Count.ToString();

    public static void RunWithDump(string dir, Action f)
    {
        RunWithDump<int>(dir, () =>
        {
            f();
            // discard return value
            return -1;
        });
    }

    public static T RunWithDump<T>(string dir, Func<T> f)
    {
        Dir = dir;
        Count = 1;
        OpenDump = true;
        Append = false;
        var result = f();
        OpenDump = false;
        return result;
    }

    public string GetMaybeDumpDir()
    {
        return ValueDumper.GetMaybeDumpDir(Dir);
    }

    protected void UpdateOrder(string root, string target, Shape shape)
    {
        using (var order = new StreamWriter(Path.Join(root, "!out_shape_list"), Append))
        {
            order.WriteLine($"{target}: {DumpUtility.SerializeShape(shape)}");
        }
    }

    protected void DumpCallParam(string target, ParameterInfo info, Action<StreamWriter> f)
    {
        var path = Path.Join(GetMaybeDumpDir(), $"{CountStr}${target}${info.Name}");
        using (var sr = new StreamWriter(path))
        {
            f(sr);
        }
    }

    protected void DumpCall(string target, Shape shape, Action<StreamWriter> f)
    {
        var path = Path.Join(GetMaybeDumpDir(), $"{CountStr}${target}");
        using (var sr = new StreamWriter(path))
        {
            f(sr);
        }

        UpdateOrder(GetMaybeDumpDir(), target, shape);
        Append = true;
        ++Count;
    }
}

public static class ValueDumper
{
    public static void DumpTensor(TensorValue tensorValue, StreamWriter writer)
    {
        var tensor = tensorValue.AsTensor();
        if (tensor.ElementType is PrimType)
        {
            var typeCode = ((PrimType)tensor.ElementType).TypeCode;
            writer.WriteLine($"type:{(int)typeCode}");
        }
        else
        {
            writer.WriteLine($"type:0");
        }

        writer.WriteLine(DumpUtility.SerializeShape(tensor.Shape));
        // todo:other type
        var dt = tensor.ElementType;
        if (dt == DataTypes.Int8 || dt == DataTypes.Int32 || dt == DataTypes.Int64)
        {
            foreach (var v in tensor.ToArray<long>())
            {
                writer.WriteLine(v);
            }
        }
        else if (dt is PrimType)
        {
            foreach (var v in tensor.ToArray<float>())
            {
                writer.WriteLine(v);
            }
        }
        else
        {
            writer.WriteLine($"{dt} NotImpl");
        }
    }

    public static void DumpTensors(Tensor[] tensorValues, StreamWriter writer)
    {
        foreach (var tensorValue in tensorValues)
        {
            DumpTensor(tensorValue, writer);
        }
    }

    public static void DumpTensor(TensorValue tensorValue, string path)
    {
        using (var sr = new StreamWriter(path))
        {
            DumpTensor(tensorValue, sr);
        }
    }

    public static void DumpTensors(TensorValue[] tensorValue, string path)
    {
        using (var sr = new StreamWriter(path))
        {
            DumpTensors(tensorValue.Select(x => x.AsTensor()).ToArray(), sr);
        }
    }

    public static string GetMaybeDumpDir(string dir)
    {
        var root = Path.Join(CompilerServices.CompileOptions.DumpDir, dir);
        if (!Directory.Exists(root))
        {
            Directory.CreateDirectory(root);
        }

        return root;
    }
}

public static class DumpUtility
{
    public static void WriteResult(string path, string data, string prefix = "")
    {
        using (var stream = new StreamWriter(path))
        {
            stream.Write(prefix);
            stream.Write(data);
        }
    }

    public static void WriteResult<T>(string path, T[] data, string prefix = "")
    {
        WriteResult(path, SerializeByColumn(data), prefix);
    }

    public static string SerializeByColumn<T>(T[] f)
    {
        return string.Join("\n", f);
    }

    public static string SerializeByRow<T>(T[] arr)
    {
        return string.Join(" ", arr);
    }

    public static string SerializeShape(int[] shape)
    {
        return $"shape:{SerializeByRow(shape)}";
    }

    public static string SerializeShape(Dimension[] dims)
    {
        return $"shape:{SerializeByRow(dims)}";
    }

    public static string SerializeShape(Shape shape) => SerializeShape(shape.ToArray());

    public static string PathJoinByCreate(string root, params string[] paths)
    {
        var path = Path.Join(new[] { root }.Concat(paths).ToArray());
        Directory.CreateDirectory(path);
        return path;
    }

    public static string SnakeName(string name)
    {
        var sb = new StringBuilder();
        bool lastCapital = true;
        bool lastIsLetter = true;
        foreach (var c in name)
        {
            var isLetter = char.IsLetter(c);
            var isCaptial = isLetter ? char.IsUpper(c) : false;
            if (!lastCapital && isCaptial && sb.Length != 0)
            {
                if (lastIsLetter || c != 'D')
                    sb.Append('_');
            }

            sb.Append(char.ToLowerInvariant(c));

            if (!lastIsLetter && c == 'D')
                sb.Append('_');

            lastCapital = isCaptial;
            lastIsLetter = isLetter;
        }

        return sb.ToString().Trim('_');
    }

    public static void WriteBinFile(string path, Tensor tensor)
    {
        using (var stream = new FileStream(Path.Join(path), FileMode.Create, FileAccess.Write, FileShare.None))
        using (var writer = new BinaryWriter(stream))
        {
            foreach (var b in tensor.BytesBuffer)
            {
                writer.Write(b);
            }
        }
    }
}

public class Counter
{
    public Counter(int count = 0)
    {
        Count = count;
    }

    private int Count;

    public T Run<T>(Func<int, T> f)
    {
        return f(Count++);
    }

    public void Run(Action<int> f)
    {
        f(Count++);
    }
}