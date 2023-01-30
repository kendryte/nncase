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
using Nncase.Transform;
using Nncase.Transform.Rules.Neutral;
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestMatMulToConv2D : TestClassBase
{
    public static IEnumerable<object[]> TestMatMulToConv2DPositiveData =>
        new[]
        {
            new object[] { new[] { 5, 4 }, new[] { 4, 6 } },
            new object[] { new[] { 1, 7 }, new[] { 7, 12 } },
        };

    [Theory]
    [MemberData(nameof(TestMatMulToConv2DPositiveData))]
    public void TestMatMulToConv2DPositive(int[] aShape, int[] bShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, aShape);
        var b = Random.Normal(DataTypes.Float32, 0, 1, 0, bShape);
        var rootPre = Math.MatMul(a, b);
        var rootPost = CompilerServices.Rewrite(
            rootPre,
            new IRewriteRule[]
            {
                new MatMulToConv2D(),
            },
            new());

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
}
