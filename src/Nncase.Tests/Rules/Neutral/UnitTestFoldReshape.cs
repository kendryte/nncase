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
    public static TheoryData<long[], long[], long[], long[]> TestReshapeBinaryConstReshapePositiveData => new()
    {
        { new long[] { 12, 77, 77 }, new long[] { 1, 12, 77, 77 }, new long[] { 1, 1, 77, 77 }, new long[] { 12, 77, 77 } },
        { new long[] { 12, 77, 77 }, new long[] { 1, 12, 77, 77 }, new long[] { 77 }, new long[] { 12, 77, 77 } },
    };

    public static IEnumerable<object[]> TestFoldNopReshapePositiveData =>
        new[]
        {
            new object[] { new long[] { 4 }, new long[] { 4 } },
            new object[] { new long[] { 2, 3 }, new long[] { 2, 3 } },
            new object[] { new long[] { 2, 4 }, new long[] { -1, 4 } },
        };

    public static IEnumerable<object[]> TestFoldNopReshapeNegativeData =>
        new[]
        {
            new object[] { new long[] { 4 }, new long[] { 2, 2 } },
            new object[] { new long[] { 2, 3 }, new long[] { 3, 2 } },
        };

    public static IEnumerable<object[]> TestFoldTwoReshapesPositiveData =>
        new[]
        {
            new object[] { new long[] { 4 }, new long[] { 2, 2 }, new long[] { 1, 4 } },
            new object[] { new long[] { 2, 4 }, new long[] { 8 }, new long[] { 4, 2 } },
        };

    public static IEnumerable<object[]> TestReshapeToTransposePositiveData =>
        new[]
        {
            new object[] { new long[] { 1, 1, 400, 192 }, new long[] { 1, 400, 1, 192 } },
            new object[] { new long[] { 1, 1, 1, 1 }, new long[] { 1, 1, 1, 1 } },
            new object[] { new long[] { 4, 4, 4, 4 }, new long[] { 4, 4, 4, 4 } },
        };

    public static IEnumerable<object[]> TestReshapeToTransposeNegativeData =>
        new[]
        {
            new object[] { new long[] { 1, 1, 4, 4 }, new long[] { 1, 1, 2, 8 } },
        };

    [Theory]
    [MemberData(nameof(TestFoldNopReshapePositiveData))]
    public void TestFoldNopReshapePositive(long[] shape, long[] newShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Reshape(a, newShape);
        TestMatched<FoldNopReshape>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestFoldNopReshapeNegativeData))]
    public void TestFoldNopReshapeNegative(long[] shape, long[] newShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Reshape(a, newShape);
        TestNotMatch<FoldNopReshape>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestFoldTwoReshapesPositiveData))]
    public void TestFoldTwoReshapesPositive(long[] shape, long[] newShape1, long[] newShape2)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Reshape(Tensors.Reshape(a, newShape1), newShape2);
        TestMatched<FoldTwoReshapes>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestReshapeToTransposePositiveData))]
    public void TestReshapeToTransposePositive(long[] shape, long[] newShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Reshape(a, newShape);
        TestMatched<ReshapeToTranspose>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestReshapeToTransposeNegativeData))]
    public void TestReshapeToTransposeNegative(long[] shape, long[] newShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Reshape(a, newShape);
        TestNotMatch<ReshapeToTranspose>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestReshapeBinaryConstReshapePositiveData))]
    public void TestReshapeBinaryConstReshapePositive(long[] inShape, long[] unsqShape, long[] constShape, long[] sqShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, inShape);
        var v0 = Tensors.Reshape(a, unsqShape);
        var v1 = Math.Binary(BinaryOp.Add, v0, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, constShape).Evaluate().AsTensor());
        var v2 = Tensors.Reshape(v1, sqShape);
        var rootPre = v2;
        TestMatched<FoldReshapeBinaryConstReshape>(rootPre);
    }
}
