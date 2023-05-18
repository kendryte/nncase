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
    /// <param name="tensorValues">tensor value.</param>
    /// <param name="writer">writer.</param>
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
    /// Dump multi tensor to dir with name i.
    /// </summary>
    /// <param name="tensorValue">tensor value.</param>
    /// <param name="dir">tensor.</param>
    public static void DumpTensors(TensorValue[] tensorValue, string dir)
    {
        Directory.CreateDirectory(dir);
        for (var i = 0; i < tensorValue.Length; i++)
        {
            using (var sr = new StreamWriter(Path.Join(dir, $"{i}.txt")))
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

    public static void WriteKmodelData(Tensor[] inputs, Tensor[] outputs, string kmodelPath, string dumpDir, bool dynamic)
    {
        Directory.CreateDirectory(dumpDir);
        BinFileUtil.WriteBinInputs(inputs, dumpDir);
        BinFileUtil.WriteBinOutputs(outputs, dumpDir);
        var kmodel_path = Path.Join(dumpDir, "test.kmodel");
        if (File.Exists(kmodel_path))
        {
            File.Delete(kmodel_path);
        }
        File.Copy(kmodelPath, kmodel_path);
        if (dynamic)
        {
            WriteKmodelDesc(inputs, outputs, dumpDir);
        }
    }

    public static void WriteKmodelDesc(Tensor[] inputs, Tensor[] outputs, string dir)
    {
        var inputStr = string.Join("\n", inputs.Select(input => string.Join(" ", input.Shape.ToValueArray())));
        var outputStr = string.Join("\n", outputs.Select(output => string.Join(" ", output.Shape.ToValueArray())));
        var content =
            $"{inputs.Length} {outputs.Length}\n{inputStr}\n{outputStr}";
        DumpUtility.WriteResult(Path.Join(dir, "kmodel.desc"), content);
    }
}

public static class BinFileUtil
{
    public static void WriteBinOutputs(Tensor[] outputs, string dir)
    {
        for (var i = 0; i < outputs.Length; i++)
        {
            DumpUtility.WriteBinFile(Path.Join(dir, $"nncase_result_{i}.bin"), outputs[i]);
        }
    }

    public static void WriteBinInputs(Tensor[] inputs, string dir)
    {
        for (var i = 0; i < inputs.Length; i++)
        {
            DumpUtility.WriteBinFile(Path.Join(dir, $"input_0_{i}.bin"), inputs[i]);
        }
    }

    public static Tensor ReadBinFile(string path, DataType dt, Shape shape)
    {
        using (var stream = new FileStream(Path.Join(path), FileMode.Open, FileAccess.Read, FileShare.None))
        using (var reader = new BinaryReader(stream))
        {
            var bytes = reader.ReadBytes(shape.Prod().FixedValue * dt.SizeInBytes);
            return Tensor.FromBytes(dt, bytes, shape);
        }
    }

    public static Tensor ReadBinFile(string path, Func<byte[], Tensor> f)
    {
        using (var stream = new FileStream(Path.Join(path), FileMode.Open, FileAccess.Read, FileShare.None))
        using (var reader = new BinaryReader(stream))
        {
            var bytes = reader.ReadBytes((int)stream.Length);
            return f(bytes);
        }
    }
}
