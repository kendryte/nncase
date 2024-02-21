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

            var layernorm = IR.F.CPU.PackedLayerNorm(packedInput, packedScale, packedBias, axis, 1e-6f, false, packedAxes, packedAxes.Select(i => padsInput[i, 1]).ToArray());

            post = SliceForPack(IR.F.CPU.Unpack(layernorm, packedAxes), shape, padsInput);
        }

        var feedDict = new Dictionary<Var, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
            { scale, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, pshape).Evaluate() },
            { bias, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, pshape).Evaluate() },
        };
        Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
    }

    private static Expr PadForPack(Expr input, int[] shape, int[] packedAxes, Expr value, out int[,] pads)
    {
        var isPadded = false;
        pads = new int[shape.Length, 2];
        foreach (var i in packedAxes)
        {
            if (shape[i] % Lanes != 0)
            {
                pads[i, 1] = MathUtility.AlignUp(shape[i], Lanes) - shape[i];
                isPadded = true;
            }
        }

        if (isPadded)
        {
            return IR.F.NN.Pad(input, pads, PadMode.Constant, value);
        }

        return input;
    }

    private static Expr SliceForPack(Expr input, int[] shape, int[,] pads)
    {
        bool isPadded = false;
        var ends = shape.ToArray();
        for (int i = 0; i < pads.GetLength(0); i++)
        {
            if (pads[i, 1] != 0)
            {
                isPadded = true;
            }
        }

        return isPadded ? IR.F.Tensors.Slice(input, Enumerable.Repeat(0, shape.Length).ToArray(), ends, shape.Length) : input;
    }
}
