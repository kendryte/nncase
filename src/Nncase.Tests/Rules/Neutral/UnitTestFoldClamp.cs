// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.F;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestFoldClamp : TransformTestBase
{
    public static IEnumerable<object[]> TestFoldNopClampPositiveData =>
        new[]
        {
            new object[] { float.NegativeInfinity, float.PositiveInfinity },
            new object[] { float.MinValue, float.MaxValue },
            new object[] { double.NegativeInfinity, double.PositiveInfinity },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    public static IEnumerable<object[]> TestFoldNopClampNegativeData =>
        new[]
        {
            new object[] { float.MinValue, float.IsNormal(10) },
            new object[] { float.IsNormal(-2), float.IsNormal(10) },
            new object[] { float.IsNormal(-2), float.MaxValue },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    [Theory]
    [MemberData(nameof(TestFoldNopClampPositiveData))]
    public void TestFoldNopCastPositive(float min, float max, int index)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var rootPre = Math.Clamp(a, min, max);
        TestMatched<FoldNopClamp>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestFoldNopClampNegativeData))]
    public void TestFoldNopCastNegative(float min, float max, int index)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var rootPre = Math.Clamp(a, min, max);
        TestNotMatch<FoldNopClamp>(rootPre);
    }
}
