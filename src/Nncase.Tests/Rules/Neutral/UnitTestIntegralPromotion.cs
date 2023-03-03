// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Transforms;
using Nncase.Tests.TestFixture;
using Tensorflow.Operations.Initializers;
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestIntegralPromotion : TestClassBase
{
    public static IEnumerable<object[]> TestIntegralPromotionPositiveData =>
        new[]
        {
            new object[] { DataTypes.Int32, DataTypes.Int64 },
            new object[] { DataTypes.Int64, DataTypes.Int32 },
        };

    public static IEnumerable<object[]> TestIntegralPromotionNegativeData =>
        new[]
        {
            new object[] { DataTypes.Int32, DataTypes.Int32 },
            new object[] { DataTypes.Int64, DataTypes.Int64 },
        };

    [Theory]
    [MemberData(nameof(TestIntegralPromotionPositiveData))]
    public async Task TestIntegralPromotionPositive(DataType aType, DataType bType)
    {
        var expr = Tensors.Cast(1, aType) + Tensors.Cast(2, bType);
        expr.InferenceType();
        var f = new Function(expr);
        var result = CompilerServices.InferenceType(f);
        Assert.False(result);
        var post = await new ShapeInferPass { Name = "TypePromotion" }.RunAsync(f, new());
        Assert.True(CompilerServices.InferenceType(post));
        Assert.Equal(Value.FromTensor(3L), ((Function)post).Body.Evaluate());
    }

    [Theory]
    [MemberData(nameof(TestIntegralPromotionNegativeData))]
    public async Task TestIntegralPromotionNegative(DataType aType, DataType bType)
    {
        var expr = Tensors.Cast(1, aType) + Tensors.Cast(2, bType);
        expr.InferenceType();
        var f = new Function(expr);
        var result = CompilerServices.InferenceType(f);
        Assert.True(result);
        var post = await new ShapeInferPass { Name = "TypePromotion" }.RunAsync(f, new());
        Assert.True(CompilerServices.InferenceType(post));
        Assert.Equal(Value.FromTensor(3L), ((Function)post).Body.Evaluate());
    }
}
