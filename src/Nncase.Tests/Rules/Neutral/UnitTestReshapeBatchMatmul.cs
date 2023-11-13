// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestReshapeBatchMatmul : TransformTestBase
{
    public static IEnumerable<object[]> TestReshapeBatchMatmulPositiveData =>
        new[]
        {
            new object[] { new[] { 2, 1, 4 }, new[] { 4, 6 } },
            new object[] { new[] { 3, 1, 7 }, new[] { 1, 7, 12 } },
            new object[] { new[] { 1, 3, 1, 7 }, new[] { 1, 1, 7, 12 } },
        };

    public static IEnumerable<object[]> TestReshapeBatchMatmulNegativeData =>
        new[]
        {
            new object[] { new[] { 2, 1, 4 }, new[] { 4, Dimension.Unknown } },
        };

    [Theory]
    [MemberData(nameof(TestReshapeBatchMatmulPositiveData))]
    public void TestReshapeBatchMatmulPositive(int[] aShape, int[] bShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, aShape);
        var b = Random.Normal(DataTypes.Float32, 0, 1, 0, bShape);
        var rootPre = Math.MatMul(a, b);
        TestMatched<ReshapeBatchMatmul>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestReshapeBatchMatmulNegativeData))]
    public void TestReshapeBatchMatmulNegative(int[] aShape, Dimension[] bShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, aShape);
        var b = new IR.Var(new IR.TensorType(DataTypes.Float32, bShape));
        var rootPre = Math.MatMul(a, b);
        TestNotMatch<ReshapeBatchMatmul>(rootPre);
    }
}
