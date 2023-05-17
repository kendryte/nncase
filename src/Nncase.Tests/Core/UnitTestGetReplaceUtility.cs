// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Security.Cryptography;
using Nncase;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using OrtKISharp;
using Xunit;
using Fx = System.Func<Nncase.IR.Expr, Nncase.IR.Expr>;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestGetReplaceUtility
{
    [Fact]
    public void TestGetReplaceUtility()
    {
        Assert.Throws<InvalidOperationException>(() => Utility.Get4DGNNEShape(new[] { 0, 1, 2, 3, 4 }));
    }

    [Fact]
    public void WithTmp4DShape_WhenGivenFunctionAndOutputShape_ShouldInsertRehsapeBeforeAndAfterCall()
    {
        var input = IR.F.Random.Normal(DataTypes.UInt8, new[] { 1, 2, 3, 4 });
        Fx inputCtor = expr => IR.F.Tensors.Flatten(expr, 1);
        var originOutShape = new[] { 1, 2, 3, 4 };

        var output = Utility.WithTmp4DShape(inputCtor, originOutShape)(input);

        Assert.Equal(input.CheckedType, output.CheckedType);
        Assert.Equal(originOutShape, output.CheckedShape.ToValueArray());
    }

    [Fact]
    public void WithTmpType_WhenGivenFunctionAndDataType_ShouldInsertCastBeforeAndAfterCall()
    {
        var input = IR.F.Random.Normal(DataTypes.UInt8, new[] { 1, 2, 3, 4 });
        Fx inputCtor = expr => IR.F.Tensors.Flatten(expr, 1);
        var dataType = DataTypes.Float32;

        var output = Utility.WithTmpType(inputCtor, dataType)(input);

        Assert.Equal(new[] { 1, 24 }, output.CheckedShape);
    }

    [Fact]
    public void WithTmpBF16_WhenGivenFunction_ShouldInsertCastBeforeAndAfterCall()
    {
        var input = IR.F.Random.Normal(DataTypes.Float32, new[] { 1, 2, 3, 4 });
        Fx inputCtor = expr => IR.F.Tensors.Flatten(expr, 1);

        var output = Utility.WithTmpBF16(inputCtor)(input);

        Assert.Equal(new[] { 1, 24 }, output.CheckedShape);
    }

    [Fact]
    public void Apply_WhenGivenFunction_ShouldApplyFunctionToInput()
    {
        var input = IR.F.Random.Normal(DataTypes.UInt8, new[] { 1, 2, 3, 4 });
        Fx inputCtor = expr =>
            IR.F.NN.Pad(expr, new[,] { { 0, 0 }, { 0, 0 }, { 1, 1 }, { 1, 1 } }, PadMode.Constant, (byte)0);

        var output = Utility.Apply(f => expr => f(expr), inputCtor)(input);

        Assert.Equal(new[] { 1, 2, 5, 6 }, output.CheckedShape);
    }
}
