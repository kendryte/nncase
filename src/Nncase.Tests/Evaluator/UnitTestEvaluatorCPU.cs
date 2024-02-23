// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Toolkit.HighPerformance;
using Nncase.IR;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.EvaluatorTest;

public sealed class UnitTestEvaluatorCPU
{
    public static int Lanes => 32;

    [Theory]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 0, new[] { 2 } })] // unrelated with axis
    [InlineData(new object[] { new[] { 32, 64, 128 }, 1, new[] { 0 } })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 2, new[] { 1 } })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 0, new[] { 0 } })] // packed on axis
    [InlineData(new object[] { new[] { 32, 64, 128 }, 1, new[] { 1 } })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 2, new[] { 2 } })]
    [InlineData(new object[] { new[] { 36, 64, 128 }, 0, new[] { 2 } })] // padded but packed not on axis
    [InlineData(new object[] { new[] { 32, 69, 128 }, 1, new[] { 0 } })]
    [InlineData(new object[] { new[] { 32, 64, 135 }, 2, new[] { 1 } })]
    [InlineData(new object[] { new[] { 36, 64, 128 }, 0, new[] { 0 } })] // padded and packed on axis
    [InlineData(new object[] { new[] { 32, 69, 128 }, 1, new[] { 1 } })]
    [InlineData(new object[] { new[] { 32, 64, 135 }, 2, new[] { 2 } })]
    public void TestPackedSoftMax(int[] shape, int axis, int[] packedAxes)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.NN.Softmax(input, axis);

        Expr post;
        {
            var packed = IR.F.CPU.Pack(PadForPack(input, shape, packedAxes, float.NegativeInfinity, out var pads), Enumerable.Repeat(Lanes, packedAxes.Length).ToArray(), packedAxes);
            var softmax = IR.F.CPU.PackedSoftMax(packed, axis, packedAxes);
            post = SliceForPack(IR.F.CPU.Unpack(softmax, packedAxes), shape, pads);
        }

        var feedDict = new Dictionary<Var, IValue>() { { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() } };
        Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
    }

    [Theory]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 1, new[] { 0 } })] // unrelated with axis
    [InlineData(new object[] { new[] { 32, 64, 128 }, 2, new[] { 1 } })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 0, new[] { 0 } })] // packed on axis
    [InlineData(new object[] { new[] { 32, 64, 128 }, 0, new[] { 1 } })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 0, new[] { 2 } })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 1, new[] { 1 } })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 1, new[] { 2 } })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 2, new[] { 2 } })]
    [InlineData(new object[] { new[] { 36, 64, 128 }, 1, new[] { 0 } })] // padded but packed not on axis
    [InlineData(new object[] { new[] { 32, 69, 128 }, 2, new[] { 1 } })]
    [InlineData(new object[] { new[] { 35, 64, 128 }, 0, new[] { 0 } })]// padded and packed on axis
    [InlineData(new object[] { new[] { 32, 60, 128 }, 0, new[] { 1 } })]
    [InlineData(new object[] { new[] { 32, 64, 199 }, 0, new[] { 2 } })]
    [InlineData(new object[] { new[] { 32, 57, 128 }, 1, new[] { 1 } })]
    [InlineData(new object[] { new[] { 32, 64, 81 }, 1, new[] { 2 } })]
    [InlineData(new object[] { new[] { 32, 64, 99 }, 2, new[] { 2 } })]
    public void TestPackedLayerNorm(int[] shape, int axis, int[] packedAxes)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pshape = shape.Skip(axis).ToArray();
        var scale = new Var(new TensorType(DataTypes.Float32, pshape));
        var bias = new Var(new TensorType(DataTypes.Float32, pshape));
        var pre = IR.F.NN.LayerNorm(axis, 1e-6f, input, scale, bias, false);

        Expr post;
        {
            var packedInput = IR.F.CPU.Pack(PadForPack(input, shape, packedAxes, 0f, out var padsInput), Enumerable.Repeat(Lanes, packedAxes.Length).ToArray(), packedAxes);

            var pAxes = packedAxes.Where(i => i >= axis).Select(i => i - axis).ToArray();
            var packedScale = PadForPack(scale, pshape, pAxes, 0f, out var padsScale);
            if (pAxes.Length > 0)
            {
                packedScale = IR.F.CPU.Pack(packedScale, Enumerable.Repeat(Lanes, pAxes.Length).ToArray(), pAxes);
            }

            var packedBias = PadForPack(bias, pshape, pAxes, 0f, out var padsBias);
            if (pAxes.Length > 0)
            {
                packedBias = IR.F.CPU.Pack(packedBias, Enumerable.Repeat(Lanes, pAxes.Length).ToArray(), pAxes);
            }

            var layernorm = IR.F.CPU.PackedLayerNorm(packedInput, packedScale, packedBias, axis, 1e-6f, false, packedAxes, padsInput);

            post = SliceForPack(IR.F.CPU.Unpack(layernorm, packedAxes), shape, padsInput);
        }

        var feedDict = new Dictionary<Var, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
            { scale, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, pshape).Evaluate() },
            { bias, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, pshape).Evaluate() },
        };
        Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
    }

    [Theory]
    [InlineData(new object[] { new[] { 12, 128, 768 }, new[] { 12, 768, 64 }, new[] { 1, 2 }, new[] { 1, 2 } })] // no broadcast, no pad
    [InlineData(new object[] { new[] { 1, 128, 768 }, new[] { 12, 768, 64 }, new[] { 1, 2 }, new[] { 1, 2 } })] // broadcast, no pad
    [InlineData(new object[] { new[] { 1, 129, 768 }, new[] { 12, 768, 64 }, new[] { 1, 2 }, new[] { 1, 2 } })] // broadcast, pad
    [InlineData(new object[] { new[] { 1, 128, 777 }, new[] { 12, 777, 64 }, new[] { 1, 2 }, new[] { 1, 2 } })] // broadcast, pad
    [InlineData(new object[] { new[] { 1, 131, 776 }, new[] { 12, 776, 64 }, new[] { 1, 2 }, new[] { 1, 2 } })] // broadcast, pad
    [InlineData(new object[] { new[] { 1, 131, 776 }, new[] { 12, 776, 58 }, new[] { 1, 2 }, new[] { 1, 2 } })] // broadcast, pad
    public void TestPackedMatMul(int[] lhsShape, int[] rhsShape, int[] lhsPackedAxes, int[] rhsPackedAxes)
    {
        var lhs = new Var(new TensorType(DataTypes.Float32, lhsShape));
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsShape));
        var pre = IR.F.Tensors.MatMul(lhs, rhs);

        Expr post;
        {
            var packedLhs = IR.F.CPU.Pack(PadForPack(lhs, lhsShape, lhsPackedAxes, 0f, out var lhsPadNums), Enumerable.Repeat(Lanes, lhsPackedAxes.Length).ToArray(), lhsPackedAxes);
            var packedRhs = IR.F.CPU.Pack(PadForPack(rhs, rhsShape, rhsPackedAxes, 0f, out var rhsPadNums), Enumerable.Repeat(Lanes, rhsPackedAxes.Length).ToArray(), rhsPackedAxes);

            var matmul = IR.F.CPU.PackedMatMul(packedLhs, packedRhs, lhsPackedAxes, lhsPadNums, rhsPackedAxes, rhsPadNums);
            post = SliceForPack(IR.F.CPU.Unpack(matmul, new[] { lhsPackedAxes[0], rhsPackedAxes[1] }), pre.CheckedShape.ToValueArray(), new[] { lhsPadNums[0], rhsPadNums[1] });
        }

        var feedDict = new Dictionary<Var, IValue>() {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate() },
        };
        Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
    }

    [Theory]
    [InlineData(new object[] { BinaryOp.Add, new[] { 1, 77, 768 }, new int[] { 1, 77, 768 }, new[] { 1, 2 }, new int[] { 1, 2 } })]
    [InlineData(new object[] { BinaryOp.Add, new[] { 12, 77, 64 }, new[] { 12, 1, 64 }, new[] { 1, 2 }, new[] { 2 } })]
    [InlineData(new object[] { BinaryOp.Mul, new[] { 12, 77, 64 }, new int[] { }, new[] { 1, 2 }, new int[] { } })]
    [InlineData(new object[] { BinaryOp.Add, new[] { 12, 77, 77 }, new int[] { 1, 77, 77 }, new[] { 1 }, new int[] { 1 } })]
    [InlineData(new object[] { BinaryOp.Add, new[] { 12, 77, 77 }, new int[] { 1, 77, 77 }, new[] { 2 }, new int[] { 2 } })]
    [InlineData(new object[] { BinaryOp.Add, new[] { 12, 77, 77 }, new int[] { 1, 77, 77 }, new[] { 1, 2 }, new int[] { 1, 2 } })]
    [InlineData(new object[] { BinaryOp.Add, new[] { 1, 77, 768 }, new int[] { 768 }, new[] { 1, 2 }, new int[] { 0 } })]
    [InlineData(new object[] { BinaryOp.Add, new[] { 1, 77, 3072 }, new int[] { 3072 }, new[] { 1, 2 }, new int[] { 0 } })]
    public void TestPackedBinary(BinaryOp op, int[] lhsShape, int[] rhsShape, int[] lhsPackedAxes, int[] rhsPackedAxes)
    {
        var lhs = new Var(new TensorType(DataTypes.Float32, lhsShape));
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsShape));
        var pre = IR.F.Tensors.MatMul(lhs, rhs);

        Expr post;
        {
            var packedLhs = IR.F.CPU.Pack(PadForPack(lhs, lhsShape, lhsPackedAxes, 0f, out var lhsPadNums), Enumerable.Repeat(Lanes, lhsPackedAxes.Length).ToArray(), lhsPackedAxes);
            var packedRhs = IR.F.CPU.Pack(PadForPack(rhs, rhsShape, rhsPackedAxes, 0f, out var rhsPadNums), Enumerable.Repeat(Lanes, rhsPackedAxes.Length).ToArray(), rhsPackedAxes);

            var binary = IR.F.CPU.PackedBinary(packedLhs, packedRhs, op, lhsPackedAxes, lhsPadNums, rhsPackedAxes, rhsPadNums);
            post = SliceForPack(IR.F.CPU.Unpack(binary, lhsPackedAxes.Length >= rhsPackedAxes.Length ? lhsPackedAxes : rhsPackedAxes), pre.CheckedShape.ToValueArray(), lhsPackedAxes.Length >= rhsPackedAxes.Length ? lhsPadNums : rhsPadNums);
        }

        var feedDict = new Dictionary<Var, IValue>() {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate() },
        };
        Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
    }

    private static Expr PadForPack(Expr input, int[] shape, int[] packedAxes, Expr value, out int[] padNums)
    {
        var isPadded = false;
        var pads = new int[shape.Length, 2];
        foreach (var i in packedAxes)
        {
            if (shape[i] % Lanes != 0)
            {
                pads[i, 1] = MathUtility.AlignUp(shape[i], Lanes) - shape[i];
                isPadded = true;
            }
        }

        padNums = new int[packedAxes.Length];
        for (int i = 0; i < packedAxes.Length; i++)
        {
            padNums[i] = pads[packedAxes[i], 1];
        }

        if (isPadded)
        {
            return IR.F.NN.Pad(input, pads, PadMode.Constant, value);
        }

        return input;
    }

    private static Expr SliceForPack(Expr input, int[] shape, int[] padNums)
    {
        bool isPadded = false;
        var ends = shape.ToArray();
        if (padNums.Any(i => i > 0))
        {
            isPadded = true;
        }

        return isPadded ? IR.F.Tensors.Slice(input, Enumerable.Repeat(0, shape.Length).ToArray(), ends, shape.Length) : input;
    }
}
