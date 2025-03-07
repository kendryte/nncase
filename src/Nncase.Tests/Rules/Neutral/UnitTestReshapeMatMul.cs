// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Collections.Generic;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Math;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestReshapeMatMul : TransformTestBase
{
    public static IEnumerable<object[]> MatMulShapeData => new[]
    {
        new object[] { new long[] { 2, 3, 7, 9 }, new long[] { 9, 7 } },
        new object[] { new long[] { 7, 9 }, new long[] { 2, 3, 9, 7 } },
        new object[] { new long[] { 3, 7 }, new long[] { 7 } },
        new object[] { new long[] { 7 }, new long[] { 7, 3 } },
        new object[] { new long[] { 2, 3, 7 }, new long[] { 7 } },
        new object[] { new long[] { 3 }, new long[] { 2, 3, 7 } },
    };

    public static IEnumerable<object[]> NopMatMulShapeData => new[]
    {
        new object[] { new long[] { 1, 3, 24, 24 }, new long[] { 3, 24, 24 } },
        new object[] { new long[] { 7, 3 }, new long[] { 3, 7 } },
    };

    [Theory]
    [MemberData(nameof(MatMulShapeData))]
    public void TestTo3D(long[] shapeA, long[] shapeB)
    {
        var lhs = Testing.Rand<float>(shapeA);
        var rhs = Testing.Rand<float>(shapeB);
        var mm = MatMul(lhs, rhs);
        TestMatched<ReshapeMatMul>(mm);
    }

    [Theory]
    [MemberData(nameof(NopMatMulShapeData))]
    public void TestNop(long[] shapeA, long[] shapeB)
    {
        var lhs = Testing.Rand<float>(shapeA);
        var rhs = Testing.Rand<float>(shapeB);
        var mm = MatMul(lhs, rhs);
        TestNotMatch<ReshapeMatMul>(mm);
    }
}
