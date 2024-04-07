// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests;

public static class DataGenerator
{
    private static readonly System.Random Rand = new();

    public static int DefaultChannel => 2;

    public static int[] DefaultShape => new[] { 3, 2, 4, 8 };

    public static IEnumerable<T> EnumValues<T>()
    {
        return Enum.GetValues(typeof(T)).Cast<T>();
    }

    public static Expr DefaultRandom()
    {
        return DefaultRandom(DataTypes.Float32, new[] { 3, DefaultChannel, 4, 8 });
    }

    public static Expr DefaultRandom(DataType dt)
    {
        return DefaultRandom(dt, DefaultShape);
    }

    public static Expr DefaultRandom(int[] shape)
    {
        return DefaultRandom(DataTypes.Float32, shape);
    }

    public static Expr DefaultRandom(DataType dt, int[] shape)
    {
        // if (dt.IsIntegral())
        // {
        //     return Testing.Rand(dt, shape);
        // }
        return
            Random.Normal(DataTypes.Float32, new Shape(shape)).Evaluate().AsTensor().CastTo(dt);
    }

    public static Expr RandomLimitOne()
    {
        return Sigmoid(Random.Normal(DataTypes.Float32, new Shape(DefaultShape)));
    }

    public static Expr RandomGNNEScalar()
    {
        return Sigmoid(
            Random.Normal(DataTypes.Float32, new Shape(1, 1, 1, 1))).Evaluate().AsTensor();
    }

    public static Expr RandomScalar()
    {
        return Sigmoid(
            Random.Normal(DataTypes.Float32, new[] { 1 })).Evaluate().AsTensor();
    }

    // nncase format DeQuantizeParam
    public static QuantParam RandomQuantParam()
    {
        var qp = new QuantParam(Rand.Next(1, 2), Rand.Next(55, 255));
        return qp with { Scale = 1 / qp.Scale };
    }

    public static Expr DefaultConv()
    {
        var input = Random.Normal(DataTypes.Float32, new[] { 1, 3, 24, 32 });
        var weights = Random.Normal(DataTypes.Float32, new[] { 16, 3, 3, 3 }).Evaluate();
        var bias = Random.Normal(DataTypes.Float32, new[] { 16 }).Evaluate();
        var stride = Tensor.From(new[] { 1, 1 }, new[] { 2 });
        var dilation = Tensor.From(new[] { 1, 1 }, new[] { 2 });
        var padding = new[,]
        {
            { 0, 1 },
            { 0, 0 },
        };

        var conv = Conv2D(input, weights.AsTensor(), bias.AsTensor(), stride, padding, dilation, PadMode.Constant, 1);
        return conv;
    }

    public static IEnumerable<object[]> ResizeModeProduct()
    {
        var v1 = EnumValues<ImageResizeMode>().Select(x => (object)x).ToArray();
        var v2 = EnumValues<ImageResizeNearestMode>().Select(x => (object)x).ToArray();
        var v3 = EnumValues<ImageResizeTransformationMode>().Select(x => (object)x).ToArray();
        var result = Product(
            new[] { v1, v2, v3 }).Select(x => (object[])x.ToArray());
        return result;
    }

    public static IEnumerable<IEnumerable<T>> Product<T>(
        this IEnumerable<IEnumerable<T>> sequences)
    {
        IEnumerable<IEnumerable<T>> emptyProduct =
            new[] { Enumerable.Empty<T>() };
        var ret = sequences.Aggregate(
            emptyProduct,
            (accumulator, sequence) =>
                from accseq in accumulator
                from item in sequence
                select accseq.Concat(new[] { item }));
        return ret;
    }

    public static IValue FromTextFile(string path)
    {
        using (var stream = new StreamReader(path))
        {
            var content = stream.ReadToEnd().Trim().Split("\n");
            var data = ParseDumpFile(content);
            if (content[1] == "tuple")
            {
                return Value.FromTensors(data.Select(ParseTensor).ToArray());
            }
            else
            {
                Assert.Equal(data.Length, 1);
                return Value.FromTensor(ParseTensor(data.First()));
            }
        }
    }

    public static Call AutoConstructor(string root, string opNameInFile, int num)
    {
        var opTy = new IR.Tensors.Broadcast().GetType().Assembly.DefinedTypes
            .First(ty => ty.Name.Contains(opNameInFile, StringComparison.OrdinalIgnoreCase));
        var op = (Op)Activator.CreateInstance(opTy)!;
        var data = new TextDataExtractor().GetFilesByOrderNum(root);
        var opdata = data[$"{num}${opNameInFile}"];
        var parameters = op.Parameters
            .Select((param, i) => opdata.First(dataPath => dataPath.Contains(param.Name, StringComparison.OrdinalIgnoreCase)))
            .Select(path => (Expr)FromTextFile(path).AsTensor()).ToArray();
        return new Call(op, parameters);
    }

    private static float ParseFloat(string s)
    {
        if (s == "inf")
        {
            return float.PositiveInfinity;
        }
        else if (s == "-inf")
        {
            return float.NegativeInfinity;
        }
        else
        {
            return float.Parse(s);
        }
    }

    private static Tensor ParseTensor(DumpData dumpData)
    {
        var (dt, shape, data) = dumpData;
        return dt switch
        {
            PointerType pointerType => throw new NotImplementedException(),
            BooleanType booleanType => Tensor.From(data.Select(x => int.Parse(x) >= 1).ToArray(), shape),
            Float16Type float16Type => Tensor.From(data.Select(x => (Half)ParseFloat(x)).ToArray(), shape),
            Float32Type float32Type => Tensor.From(data.Select(x => ParseFloat(x)).ToArray(), shape),
            Float64Type float64Type => Tensor.From(data.Select(x => double.Parse(x)).ToArray(), shape),
            Int16Type int16Type => Tensor.From(data.Select(x => short.Parse(x)).ToArray(), shape),
            Int32Type int32Type => Tensor.From(data.Select(x => int.Parse(x)).ToArray(), shape),
            Int64Type int64Type => Tensor.From(data.Select(x => long.Parse(x)).ToArray(), shape),
            Int8Type int8Type => Tensor.From(data.Select(x => sbyte.Parse(x)).ToArray(), shape),
            BFloat16Type bFloat16Type => throw new NotImplementedException(),
            UInt16Type uInt16Type => Tensor.From(data.Select(x => ushort.Parse(x)).ToArray(), shape),
            UInt32Type uInt32Type => Tensor.From(data.Select(x => uint.Parse(x)).ToArray(), shape),
            UInt64Type uInt64Type => Tensor.From(data.Select(x => ulong.Parse(x)).ToArray(), shape),
            UInt8Type uInt8Type => Tensor.From(data.Select(x => byte.Parse(x)).ToArray(), shape),
            Utf8CharType utf8CharType => throw new NotImplementedException(),
            PrimType primType => throw new NotImplementedException(),
            QuantParamType quantParamType => throw new NotImplementedException(),
            ValueType valueType => throw new NotImplementedException(),
            _ => throw new ArgumentOutOfRangeException(nameof(dumpData), $"Invalid dataype: {dt}."),
        };
    }

    // todo: support tuple parse
    // todo: support Evaluator dump result

    // Runtime Format
    // input format:
    // datatype: xxx
    // shape: x x x x
    // data[0]
    // data[1]
    // ...
    // data[n]

    // output format:
    // op
    // datatype: xxx
    // shape: x x x x
    // data[0]
    // data[1]
    // ...
    // data[n]
    private static (DataType DataType, int[] Shape, string[] Data, int EndIndex) ParseDumpFile(string[] content, int baseIndex)
    {
        var dtIndex = baseIndex;
        var shapeIndex = baseIndex + 1;
        var data = content[(shapeIndex + 1)..];
        var end = data.FirstOrDefault(d => d.StartsWith("type"));
        var endIndex = end is null ? data.Length : Array.IndexOf(data, end);
        var endIndexInContent = endIndex + shapeIndex + 1;
        return (ParseDataType(content[dtIndex]), ParseShape(content[shapeIndex]), data[..endIndex], endIndexInContent);
    }

    private static DumpData[] ParseDumpFile(string[] content)
    {
        int baseIndex = 0;

        // result
        if (!content[0].StartsWith("type"))
        {
            baseIndex = 1;
        }

        if (content[1] == "tuple")
        {
            baseIndex = 2;
        }

        var result = new List<DumpData>();
        while (true)
        {
            var (dt, shape, data, endIndex) = ParseDumpFile(content, baseIndex);
            baseIndex = endIndex;
            result.Add(new DumpData(dt, shape, data));
            if (endIndex == content.Length)
            {
                return result.ToArray();
            }
        }
    }

    // format
    // shape: x x x x
    private static int[] ParseShape(string shapeStr)
    {
        var s = shapeStr.TrimEnd().Split(":")[1];
        if (s == "scalar")
        {
            return Array.Empty<int>();
        }

        return s.Split(" ").Select(x => int.Parse(x)).ToArray();
    }

    private static DataType ParseDataType(string dt) => DataType.FromTypeCode((Runtime.TypeCode)int.Parse(dt.Split(":")[1]));

    private record DumpData(DataType Dt, int[] Shape, string[] Data)
    {
    }
}
