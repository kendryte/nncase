// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.EvaluatorTest;

public sealed class UnitTestEvaluatorCPU
{
    public static int Lanes => 32;

    [Theory]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 2, new[] { 0 } })]
    // [InlineData(new object[] { new[] { 32, 64, 128 }, 1, new[] { 2 } })]
    // [InlineData(new object[] { new[] { 32, 64, 128 }, 1, new[] { 1 } })]
    public void TestPackedSoftMax(int[] shape, int axis, int[] packedAxes)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.NN.Softmax(input, axis);

        var post = SliceForPack(
            IR.F.CPU.PackedSoftMax(IR.F.CPU.Pack(PadForPack(input, shape, packedAxes, float.NegativeInfinity, out var pads), Enumerable.Repeat(Lanes, packedAxes.Length).ToArray(), packedAxes), axis, packedAxes), shape, pads);

        var feedDict = new Dictionary<Var, IValue>() { { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() } };
        Assert.Equal(pre.Evaluate(feedDict), post.Evaluate(feedDict));
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
