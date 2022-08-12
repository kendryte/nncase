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
    
    protected void UpdateOrder(string root, string target)
    {
        using (var order = new StreamWriter(Path.Join(root, "order"), Append))
        {
            order.WriteLine(target);
        }
    }

    protected void DumpCallParam(string target, ParameterInfo info, Action<StreamWriter> f)
    {
        var path = Path.Join(GetMaybeDumpDir(), CountStr + target + $"_param_{info.Name}");
        using (var sr = new StreamWriter(path))
        {
            f(sr);
        }
    }
    
    protected void DumpCall(string target, Action<StreamWriter> f)
    {
        var path = Path.Join(GetMaybeDumpDir(), CountStr + target);
        using (var sr = new StreamWriter(path))
        {
            f(sr);
        }
        UpdateOrder(GetMaybeDumpDir(), target);
        Append = true;
        ++Count;
    }
}

public static class ValueDumper
{
    public static void DumpTensor(TensorValue tensorValue, StreamWriter writer)
    {
        var tensor = tensorValue.AsTensor();
        writer.WriteLine(tensor.Shape.ToString());
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
            stream.WriteLine(prefix);
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
    
    public static string SerializeByRow<T>(T[] f)
    {
        return string.Join(" ", f);
    }

    public static string SerializeShape(int[] shape)
    {
        return $"Shape:[{SerializeByRow(shape)}]";
    }

    public static string PathJoinByCreate(string root, params string[] paths)
    {
        var path = Path.Join(new[]{root}.Concat(paths).ToArray());
        Directory.CreateDirectory(path);
        return path;
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