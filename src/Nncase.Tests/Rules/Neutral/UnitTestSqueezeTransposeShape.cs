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
using Nncase.IR.Math;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;
using Tensors = Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestSqueezeTransposeShape : TransformTestBase
{
    public static IEnumerable<object[]> TestSqueezeTransposeShapePosivateData =>
        new[]
        {
            new object[] { new[] { 1, 2, 4, 8, 3 }, new[] { 0, 2, 1, 3, 4 } },
            new object[] { new[] { 1, 2, 4, 8, 3 }, new[] { 0, 2, 3, 4, 1 } },
            new object[] { new[] { 1, 2, 4, 8, 3, 5, 3, 5 }, new[] { 0, 1, 3, 4, 2, 5, 6, 7 } },
            new object[] { new[] { 1, 2, 4, 8, 3, 5, 3, 5 }, new[] { 0, 1, 2, 5, 3, 4, 6, 7 } },
        };

    public static IEnumerable<object[]> TestSqueezeTransposeShapeNegativeData =>
        new[]
        {
            new object[] { new[] { 1, 2, 4 }, new[] { 0, 2, 1 } },
            new object[] { new[] { 1, 2, 4, 8 }, new[] { 0, 2, 1, 3 } },
            new object[] { new[] { 1, 2, 4, 8, 3 }, new[] { 0, 2, 1, 4, 3 } },
            new object[] { new[] { 1, 2, 4, 8, 3 }, new[] { 0, 4, 2, 1, 3 } },
            new object[] { new[] { 1, 2, 4, 8, 3, 5, 3, 5 }, new[] { 0, 1, 3, 4, 2, 6, 5, 7 } },
            new object[] { new[] { 1, 2, 4, 8, 3, 5, 3, 5 }, new[] { 7, 5, 1, 3, 0, 2, 4, 6 } },
        };

    [Theory]
    [MemberData(nameof(TestSqueezeTransposeShapePosivateData))]
    public void TestSqueezeTransposeShapePositivate(int[] shape, int[] perm)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Transpose(a, perm);
        TestMatched<SqueezeTransposeShape>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestSqueezeTransposeShapeNegativeData))]
    public void TestSqueezeTransposeShapeNegative(int[] shape, int[] perm)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Transpose(a, perm);
        TestNotMatch<SqueezeTransposeShape>(rootPre);
    }
}

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestSqueezeBinaryShape : TransformTestBase
{
    public static IEnumerable<object[]> TestSqueezeBinaryShapePosivateData =>
        new[]
        {
            new object[] { new[] { 1 }, new[] { 1, 2, 4, 8, 3 } },
            new object[] { new[] { 1, 2, 4, 8, 3 }, new[] { 1 } },
            new object[] { new[] { 1, 2, 4, 8, 3 }, new[] { 1, 1, 4, 1, 1 } },
            new object[] { new[] { 1, 2, 4, 8, 3 }, new[] { 1, 1, 4, 8, 1 } },
            new object[] { new[] { 1, 2, 4, 8, 3 }, new[] { 1, 2, 4, 8, 3 } },
            new object[] { new[] { 1, 2, 1, 8, 1 }, new[] { 3, 1, 6, 1, 1 } },
            new object[] { new[] { 1, 2, 4, 1, 3, 1, 3 }, new[] { 1, 2, 4, 1, 1, 5, 1 } },
            new object[] { new[] { 2, 3, 4, 8, 3, 5, 3, 5 }, new[] { 2, 3, 4, 8, 1, 5, 3, 5 } },
        };

    public static IEnumerable<object[]> TestSqueezeBinaryShapeNegativeData =>
        new[]
        {
            new object[] { new[] { 2 }, new[] { 2 } },
            new object[] { new[] { 1, 2 }, new[] { 1 } },
            new object[] { new[] { 1, 2 }, new[] { 1, 2 } },
            new object[] { new[] { 2 }, new[] { 1 } },
            new object[] { new[] { 1, 2, 4 }, new[] { 1 } },
            new object[] { new[] { 1, 2, 4 }, new[] { 1, 2, 1 } },
            new object[] { new[] { 1, 2, 4 }, new[] { 1, 2, 4 } },
            new object[] { new[] { 1, 2, 1, 5, 8 }, new[] { 3, 1, 4, 1, 8 } },
            new object[] { new[] { 2, 3, 4, 8, 3, 5, 3, 5 }, new[] { 2, 1, 4, 8, 1, 5, 3, 5 } },
        };

    [Theory]
    [MemberData(nameof(TestSqueezeBinaryShapePosivateData))]
    public void TestSqueezeBinaryShapePositivate(int[] lShape, int[] rShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, lShape);
        var b = Random.Normal(DataTypes.Float32, 0, 1, 0, rShape);
        var rootPre = Math.Binary(BinaryOp.Add, a, b);
        TestMatched<SqueezeBinaryShape>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestSqueezeBinaryShapeNegativeData))]
    public void TestSqueezeBinaryShapeNegative(int[] lShape, int[] rShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, lShape);
        var b = Random.Normal(DataTypes.Float32, 0, 1, 0, rShape);
        var rootPre = Math.Binary(BinaryOp.Add, a, b);
        TestNotMatch<SqueezeBinaryShape>(rootPre);
    }
}
