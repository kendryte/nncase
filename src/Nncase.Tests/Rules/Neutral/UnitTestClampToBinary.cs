// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Toolkit.HighPerformance;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestClampToBinary : TransformTestBase
{
    public static IEnumerable<object[]> ClampToBinaryPositiveData =>
        new[]
        {
            new object[] { new[] { 1, 32, 24, 24 }, new[] { float.MinValue }, new[] { float.MaxValue }, },
            new object[] { new[] { 1, 32, 24, 24 }, new[] { -1.0f }, new[] { 1.0f }, },
            new object[] { new[] { 1, 32, 24, 24 }, new[] { float.MinValue }, new[] { 1.0f }, },
        };

    [Theory]
    [MemberData(nameof(ClampToBinaryPositiveData))]
    public void TestCombineClampPositive(int[] inputShape, float[] min, float[] max)
    {
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, inputShape);
        Expr rootPre;
        {
            var v0 = input;
            var v1 = IR.F.Math.Clamp(v0, min, max);
            rootPre = v1;
        }

        TestMatched<ClampToBinary>(rootPre);
    }
}
