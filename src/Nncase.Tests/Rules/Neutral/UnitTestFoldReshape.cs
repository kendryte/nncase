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
public class UnitTestFoldReshape : TransformTestBase
{
    public static IEnumerable<object[]> TestFoldNopReshapePositiveData =>
        new[]
        {
            new object[] { new[] { 4 }, new[] { 4 } },
            new object[] { new[] { 2, 3 }, new[] { 2, 3 } },
            new object[] { new[] { 2, 4 }, new[] { -1, 4 } },
        };

    public static IEnumerable<object[]> TestFoldNopReshapeNegativeData =>
        new[]
        {
            new object[] { new[] { 4 }, new[] { 2, 2 } },
            new object[] { new[] { 2, 3 }, new[] { 3, 2 } },
        };

    public static IEnumerable<object[]> TestFoldTwoReshapesPositiveData =>
        new[]
        {
            new object[] { new[] { 4 }, new[] { 2, 2 }, new[] { 1, 4 } },
            new object[] { new[] { 2, 4 }, new[] { 8 }, new[] { 4, 2 } },
        };

    public static IEnumerable<object[]> TestReshapeToTransposePositiveData =>
        new[]
        {
            new object[] { new[] { 1, 1, 400, 192 }, new[] { 1, 400, 1, 192 } },
            new object[] { new[] { 1, 1, 1, 1 }, new[] { 1, 1, 1, 1 } },
            new object[] { new[] { 4, 4, 4, 4 }, new[] { 4, 4, 4, 4 } },
        };

    public static IEnumerable<object[]> TestReshapeToTransposeNegativeData =>
        new[]
        {
            new object[] { new[] { 1, 1, 4, 4 }, new[] { 1, 1, 2, 8 } },
        };

    [Theory]
    [MemberData(nameof(TestFoldNopReshapePositiveData))]
    public void TestFoldNopReshapePositive(int[] shape, int[] newShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Reshape(a, newShape);
        TestMatched<FoldNopReshape>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestFoldNopReshapeNegativeData))]
    public void TestFoldNopReshapeNegative(int[] shape, int[] newShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Reshape(a, newShape);
        TestNotMatch<FoldNopReshape>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestFoldTwoReshapesPositiveData))]
    public void TestFoldTwoReshapesPositive(int[] shape, int[] newShape1, int[] newShape2)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Reshape(Tensors.Reshape(a, newShape1), newShape2);
        TestMatched<FoldTwoReshapes>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestReshapeToTransposePositiveData))]
    public void TestReshapeToTransposePositive(int[] shape, int[] newShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Reshape(a, newShape);
        TestMatched<ReshapeToTranspose>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestReshapeToTransposeNegativeData))]
    public void TestReshapeToTransposeNegative(int[] shape, int[] newShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Reshape(a, newShape);
        TestNotMatch<ReshapeToTranspose>(rootPre);
    }
}
