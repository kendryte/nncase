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
        new object[] { new[] { 2, 3, 7, 9 }, new[] { 9, 7 } },
        new object[] { new[] { 7, 9 }, new[] { 2, 3, 9, 7 } },
        new object[] { new[] { 3, 7 }, new[] { 7 } },
        new object[] { new[] { 7 }, new[] { 7, 3 } },
        new object[] { new[] { 2, 3, 7 }, new[] { 7 } },
        new object[] { new[] { 3 }, new[] { 2, 3, 7 } },
    };

    public static IEnumerable<object[]> NopMatMulShapeData => new[]
    {
        new object[] { new[] { 1, 3, 24, 24 }, new[] { 3, 24, 24 } },
        new object[] { new[] { 7, 3 }, new[] { 3, 7 } },
    };

    [Theory]
    [MemberData(nameof(MatMulShapeData))]
    public void TestTo3D(int[] shapeA, int[] shapeB)
    {
        var lhs = Testing.Rand<float>(shapeA);
        var rhs = Testing.Rand<float>(shapeB);
        var mm = MatMul(lhs, rhs);
        TestMatched<ReshapeMatMul>(mm);
    }

    [Fact]
    public void TestNop()
    {
        var lhs = Testing.Rand<float>(1, 3, 24, 24);
        var rhs = Testing.Rand<float>(3, 24, 24);
        var mm = MatMul(lhs, rhs);
        TestNotMatch<ReshapeMatMul>(mm);
    }
}
