// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using OrtKISharp;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestGetReplaceUtility
{
    [Fact]
    public void TestGetReplaceUtility()
    {
        var input = OrtKI.Random(1, 4, 5, 5);
        var weight = OrtKI.Random(8, 4, 3, 3);
        var bias = OrtKI.Random(8);
        var expr = IR.F.NN.Conv2D(
            input.ToTensor(),
            weight.ToTensor(),
            bias.ToTensor(),
            stride: new[] { 1, 1 },
            padding: Tensor.From<int>(new int[] { 1, 1, 1, 1 }, new[] { 2, 2 }),
            dilation: new[] { 1, 1 },
            PadMode.Constant,
            1);
        var a = Utility.WithTmpBF16(_ => expr);
        var b = Utility.WithTmpType(_ => expr, DataTypes.Float32);
        var c = Utility.WithTmp4DShape(_ => expr, new[] { 1, 1, 1, 1 });
        var d = Utility.Get4DGNNEShape(new[] { 0, 1, 2, 3 });
        Assert.Throws<InvalidOperationException>(() => Utility.Get4DGNNEShape(new[] { 0, 1, 2, 3, 4 }));
    }
}
