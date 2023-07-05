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
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestMatMulToConv2D : TransformTestBase
{
    public static IEnumerable<object[]> TestMatMulToConv2DPositiveData =>
        new[]
        {
            new object[] { 0, new[] { 5, 4 }, new[] { 4, 6 } },
            new object[] { 1, new[] { 1, 7 }, new[] { 7, 12 } },
        };

    [Theory]
    [MemberData(nameof(TestMatMulToConv2DPositiveData))]
    public void TestMatMulToConv2DPositive(int count, int[] aShape, int[] bShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, aShape);
        var b = Random.Normal(DataTypes.Float32, 0, 1, 0, bShape).Evaluate();
        var rootPre = Math.MatMul(a, b.AsTensor());
        TestMatched<MatMulToConv2D>(rootPre);
    }
}

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestBroadcastMatMulToConv2D : TransformTestBase
{
    public static IEnumerable<object[]> TestBroadcastMatMulToConv2DPositiveData =>
        new[]
        {
            new object[] { new[] { 3, 5, 4 }, new[] { 4, 6 } },
            new object[] { new[] { 6, 1, 7 }, new[] { 7, 12 } },
        };

    [Theory]
    [MemberData(nameof(TestBroadcastMatMulToConv2DPositiveData))]
    public void TestBroadcastMatMulToConv2DPositive(int[] aShape, int[] bShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, aShape);
        var b = Random.Normal(DataTypes.Float32, 0, 1, 0, bShape).Evaluate().AsTensor();
        var rootPre = Math.MatMul(a, b);
        TestMatched<BroadcastMatMulToConv2D>(rootPre);
    }
}

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestSplitBatchMatMul : TransformTestBase
{
    public static IEnumerable<object[]> SplitBatchMatMulPositiveData =>
        new[]
        {
            new object[] { 0, new[] { 3, 5, 4 }, new[] { 3, 4, 6 } },
            new object[] { 1, new[] { 6, 1, 7 }, new[] { 6, 7, 12 } },
        };

    [Theory]
    [MemberData(nameof(SplitBatchMatMulPositiveData))]
    public void TestSplitBatchMatMulPositive(int count, int[] aShape, int[] bShape)
    {
        SetupTestMethod(true);
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, aShape);
        var b = Random.Normal(DataTypes.Float32, 0, 1, 0, bShape).Evaluate().AsTensor();
        var rootPre = Math.MatMul(a, b);
        TestMatched<SplitBatchMatMul>(rootPre);
    }
}

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestBroadcastMatMul : TransformTestBase
{
    public static IEnumerable<object[]> BroadcastMatMulPositiveData =>
        new[]
        {
            new object[] { 1, new[] { 2, 6, 1, 7 }, new[] { 1, 6, 7, 12 } },
            new object[] { 1, new[] { 3, 2, 6, 1, 7 }, new[] { 1, 1, 6, 7, 12 } },
        };

    [Theory]
    [MemberData(nameof(BroadcastMatMulPositiveData))]
    public void TestBroadcastMatMulPositive(int count, int[] aShape, int[] bShape)
    {
        SetupTestMethod(true);
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, aShape);
        var b = Random.Normal(DataTypes.Float32, 0, 1, 0, bShape).Evaluate().AsTensor();
        var rootPre = IR.F.Tensors.Reshape(Math.MatMul(a, b), new int[] { -1, aShape[^2], bShape[^1] });
        TestMatched<BroadcastMatMul>(rootPre);
    }
}
