// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Globalization;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Xunit;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestFoldQuant : TransformTestBase
{
    public static TheoryData<int, bool, int[], DataType, QuantParam, DataType, QuantParam, DataType> FoldQuantDequantData => new()
    {
        { 0, true, new[] { 1, 2, 3 }, DataTypes.Float32, new QuantParam(0, 0.0474f), DataTypes.UInt8, new QuantParam(0, 0.0474f), DataTypes.Float32 },
        { 1, false, new[] { 1, 2, 3, 4 }, DataTypes.Float32, new QuantParam(0, 0.043f), DataTypes.UInt8, new QuantParam(0, 0.333f), DataTypes.Float32 },
        { 2, false, new[] { 1, 2, 3, 4 }, DataTypes.Float32, new QuantParam(0, 0.043f), DataTypes.UInt8, new QuantParam(0, 0.043f), DataTypes.Float16 },
    };

    public static TheoryData<int, bool, int[], DataType, QuantParam, DataType, QuantParam, DataType> FoldDequantQuantData => new()
    {
        { 0, true, new[] { 1, 2, 3 }, DataTypes.UInt8, new QuantParam(0, 0.0474f), DataTypes.Float32, new QuantParam(0, 0.0474f), DataTypes.UInt8 },
        { 1, false, new[] { 1, 2, 3, 4 }, DataTypes.UInt8, new QuantParam(0, 0.043f), DataTypes.UInt8, new QuantParam(0, 0.333f), DataTypes.UInt8 },
        { 2, false, new[] { 1, 2, 3, 4 }, DataTypes.UInt8, new QuantParam(0, 0.043f), DataTypes.UInt8, new QuantParam(0, 0.043f), DataTypes.Int8 },
    };

    [Theory]
    [MemberData(nameof(FoldQuantDequantData))]
    public void TestFoldQuantDequant(int count, bool is_pos, int[] shape, DataType input_dtype, QuantParam q_param, DataType quant_type, QuantParam deq_param, DataType dequant_type)
    {
        using var dumpScope = new DumpScope($"{count}");
        var pre = IR.F.Math.Dequantize(IR.F.Math.Quantize(Random.Uniform(DataTypes.Float32, 13, 0, 1, shape), q_param, quant_type), deq_param, dequant_type);
        if (is_pos)
        {
            TestMatched<FoldQuantDeQuant>(pre);
        }
        else
        {
            TestNotMatch<FoldQuantDeQuant>(pre);
        }
    }

    [Theory]
    [MemberData(nameof(FoldDequantQuantData))]
    public void TestFoldDequantQuant(int count, bool is_pos, int[] shape, DataType input_dtype, QuantParam deq_param, DataType dequant_type, QuantParam q_param, DataType quant_type)
    {
        using var dumpScope = new DumpScope($"{count}");
        var pre = IR.F.Math.Quantize(IR.F.Math.Dequantize(Random.Normal(input_dtype, 0, 1, 0, shape), deq_param, dequant_type), q_param, quant_type);
        if (is_pos)
        {
            TestMatched<FoldDeQuantQuant>(pre);
        }
        else
        {
            TestNotMatch<FoldDeQuantQuant>(pre);
        }
    }
}
