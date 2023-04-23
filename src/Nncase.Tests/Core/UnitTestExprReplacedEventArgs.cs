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

public sealed class UnitTestExprReplacedEventArgs
{
    [Fact]
    public void TestExprReplacedEventArgs()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var alpha = 0.8F;
        var original = IR.F.NN.Celu(input.ToTensor(), alpha);
        var replace = IR.F.NN.Elu(input.ToTensor(), alpha);
        var exprReplacedEventArgs = new ExprReplacedEventArgs(original, replace);
        Assert.Equal(original, exprReplacedEventArgs.Original);
        Assert.Equal(replace, exprReplacedEventArgs.Replace);
    }
}
