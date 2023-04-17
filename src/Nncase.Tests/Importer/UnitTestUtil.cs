// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Passes.Transforms;
using Nncase.Tests.TestFixture;
using OrtKISharp;
using Xunit;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Tests.ImporterTest;

public class UnitTestUtil
{
    [Fact]
    public void TestPadTranslate()
    {
        var pads = IR.F.NN.Pad(OrtKI.Random(1, 2, 4, 8).ToTensor(), new[,] { { 0, 1, 0, 1 }, { 1, 0, 1, 0 } }, PadMode.Constant, 0f);
        Assert.Equal(
            IR.F.Tensors.Transpose(IR.F.Tensors.Reshape(pads, new[] { -1, 2 }), new[] { 1, 0 }),
            Util.PadTranslate(pads));
    }

    [Fact]
    public void TestZeroTensor()
    {
        Assert.Equal(new TensorConst(Tensor.From<int>(new[] { 0 })), Util.ZeroTensor());
    }

    [Fact]
    public void TestGetPaddings()
    {
        var input = OrtKI.Random(1, 2, 4, 8).ToTensor();
        var weights = OrtKI.Random(3, 3, 2, 2).ToTensor();
        var stride = new long[] { 1, 1, 1, 1 };
        var dilation = new long[] { 1, 1 };
        var expr = Util.GetPaddings(input, weights, stride, dilation, true);

        var (inH, inW) = Util.GetHW(input);
        var (fH, fW) = Util.GetHW(weights);
        var padH = Util.GetWindowedPadding(inH, fH, (int)stride[0], (int)dilation[0], true);
        var padW = Util.GetWindowedPadding(inW, fW, (int)stride[1], (int)dilation[1], true);
        var expect = Util.ConcatPadding(padH, padW);
        Assert.Equal(expect, expr);
    }

    [Fact]
    public void TestComputeSplit()
    {
        var input = OrtKI.Random(1, 2, 4, 8).ToTensor();
        var outputSize = 4L;
        var axis = -1L;
        var expr = Util.ComputeSplit(input, outputSize, axis);

        var expect = IR.F.Tensors.Expand(Util.ShapeIndex(input, (int)axis) / outputSize, IR.F.Tensors.Stack(new Tuple(outputSize), 0));
        Assert.Equal(expr, expect);
    }
}
