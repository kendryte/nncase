// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestFoldNopReduce : TransformTestBase
{
    public static IEnumerable<object[]> ShouldMatchData => new[]
    {
        new object[] { ReduceOp.Mean },
        new object[] { ReduceOp.Prod },
        new object[] { ReduceOp.Sum },
        new object[] { ReduceOp.Min },
        new object[] { ReduceOp.Max },
    };

    public static IEnumerable<object[]> ShouldNotMatchData => new[]
    {
        new object[] { new[] { 3 } },
        new object[] { new[] { 1, 3 } },
        new object[] { new[] { 1, 3, 4 } },
        new object[] { new[] { 1, 2, 3, 4 } },
    };

    [Theory]
    [MemberData(nameof(ShouldMatchData))]
    public void ShouldMatch(ReduceOp op)
    {
        TestMatched<FoldNopReduce>(Reduce(op, new[] { 3 }, new[] { 0 }, 0, true));
        TestMatched<FoldNopReduce>(Reduce(op, new[] { 3 }, new[] { 0 }, 0, false));
    }

    [Theory]
    [MemberData(nameof(ShouldNotMatchData))]
    public void ShouldNotMatch(int[] shape)
    {
        TestNotMatch<FoldNopReduce>(Reduce(ReduceOp.Mean, Testing.Rand<int>(shape), new[] { 0 }, 0, false));
    }
}
