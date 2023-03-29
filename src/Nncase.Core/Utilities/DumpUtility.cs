// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text;
using Nncase.IR;

namespace Nncase.Utilities;

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

    /// <summary>
    /// Dump multi tensor to single file.
    /// </summary>
    /// <param name="tensorValues"></param>
    /// <param name="writer"></param>
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

    /// <summary>
    /// Dump multi tensor to dir with name i;
    /// </summary>
    /// <param name="tensorValue"></param>
    /// <param name="dir"></param>
    public static void DumpTensors(TensorValue[] tensorValue, string dir)
    {
        Directory.CreateDirectory(dir);
        for (var i = 0; i < tensorValue.Length; i++)
        {
            using (var sr = new StreamWriter(Path.Join(dir, "{i}.txt")))
            {
                DumpTensor(tensorValue[i], sr);
            }
        }
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
                {
                    sb.Append('_');
                }
            }

            sb.Append(char.ToLowerInvariant(c));

            if (!lastIsLetter && c == 'D')
            {
                sb.Append('_');
            }

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
