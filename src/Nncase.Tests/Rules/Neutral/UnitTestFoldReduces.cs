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
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestFoldReduce : TransformTestBase
{
    public static IEnumerable<object[]> TestFoldTwoReducesPositiveData =>
        new[] {
            new object[] { new[] { 1, 3, 16, 16 }, new[] { -1 }, new[] { -1 }, false, false, 0f, 0f, ReduceOp.Mean, ReduceOp.Mean },
            new object[] { new[] { 1, 4, 16, 20 }, new[] { 3 }, new[] { -2 }, true, true, 0f, 0f, ReduceOp.Sum, ReduceOp.Sum },
            new object[] { new[] { 1, 5, 16, 15 }, new[] { -1 }, new[] { 2 }, false, false, 0f, 0f, ReduceOp.Mean, ReduceOp.Mean },
            new object[] { new[] { 1, 6, 16, 11 }, new[] { 3 }, new[] { 2 }, true, true, 0f, 0f, ReduceOp.Sum, ReduceOp.Sum },
            new object[] { new[] { 1, 6, 16, 11 }, new[] { 3 }, new[] { -2 }, true, true, 0f, 0f, ReduceOp.Sum, ReduceOp.Sum },
        };

    public static IEnumerable<object[]> TestFoldTwoReducesNegativeData =>
        new[] {
            new object[] { new[] { 1, 3, 16, 16 }, new[] { -1 }, new[] { -1 }, false, false, 0f, 0f, ReduceOp.Sum, ReduceOp.Mean },
            new object[] { new[] { 1, 4, 16, 20 }, new[] { 3 }, new[] { -2 }, true, true, 0f, 1f, ReduceOp.Sum, ReduceOp.Sum },
            new object[] { new[] { 1, 5, 16, 15 }, new[] { -1 }, new[] { 2 }, false, true, 0f, 0f, ReduceOp.Mean, ReduceOp.Mean },
            new object[] { new[] { 1, 6, 16, 11 }, new[] { 3 }, new[] { 3 }, true, true, 0f, 0f, ReduceOp.Sum, ReduceOp.Sum },
            new object[] { new[] { 1, 6, 16, 11 }, new[] { 2 }, new[] { 3 }, true, true, 0f, 0f, ReduceOp.Sum, ReduceOp.Sum },
            new object[] { new[] { 1, 6, 16, 11 }, new[] { 2, 3 }, new[] { 3 }, true, true, 0f, 0f, ReduceOp.Sum, ReduceOp.Sum },
        };

    [Theory]
    [MemberData(nameof(TestFoldTwoReducesPositiveData))]
    public void TestFoldTwoReducesPositive(int[] shape, int[] axis1, int[] axis2, bool keepDims1, bool keepDims2, float initialValue1, float initialValue2, ReduceOp reduceOp1, ReduceOp reduceOp2)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Reduce(reduceOp2, Tensors.Reduce(reduceOp1, a, axis1, initialValue1, keepDims1), axis2, initialValue2, keepDims2);
        TestMatched<FoldTwoReduce>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestFoldTwoReducesNegativeData))]
    public void TestFoldTwoReducesNegative(int[] shape, int[] axis1, int[] axis2, bool keepDims1, bool keepDims2, float initialValue1, float initialValue2, ReduceOp reduceOp1, ReduceOp reduceOp2)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Reduce(reduceOp2, Tensors.Reduce(reduceOp1, a, axis1, initialValue1, keepDims1), axis2, initialValue2, keepDims2);
        TestNotMatch<FoldTwoReduce>(rootPre);
    }
}
