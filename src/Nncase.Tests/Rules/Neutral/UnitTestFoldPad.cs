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
using Nncase.Transform;
using Nncase.Transform.Rules.Neutral;
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestFoldPad : TestFixture.UnitTestFixtrue
{
    public static IEnumerable<object[]> TestFoldNopPadPositiveData =>
        new[]
        {
            new object[] { new[] { 1 }, new[,] { { 0, 0 } } },
            new object[]
            {
                new[] { 1, 1 }, new[,]
            {
                { 0, 0 },
                { 0, 0 }
            },
            },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    public static IEnumerable<object[]> TestFoldNopPadNegativeData =>
        new[]
        {
            new object[] { new[] { 1 }, new[,] { { 1, 0 } } },
            new object[]
            {
                new[] { 1, 1 }, new[,]
            {
                { 0, 1 },
                { 2, 0 }
            },
            },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    public static IEnumerable<object[]> TestFoldTwoPadsPositiveData =>
        new[]
        {
            new object[] { new[] { 1 }, new[,] { { 0, 1 } }, new[,] { { 2, 0 } } },
            new object[]
            {
                new[] { 1, 1 }, new[,]
            {
                { 0, 1 },
                { 1, 0 },
            }, new[,]
            {
                { 1, 3 },
                { 1, 2 }
            },
            },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    public static IEnumerable<object[]> TestFoldTwoPadsNegativeData =>
       new[]
       {
            new object[] { new[] { 1 }, new[,] { { 0, 1 } }, 1.0f, new[,] { { 2, 0 } }, 2.0f },
            new object[]
            {
                new[] { 1, 1 }, new[,]
            {
                { 0, 1 },
                { 1, 0 },
            }, 1.0f, new[,]
            {
                { 1, 3 },
                { 1, 2 },
            }, 0.0f,
            },
       }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    [Theory]
    [MemberData(nameof(TestFoldNopPadPositiveData))]
    public void TestFoldNopPadPositive(int[] shape, int[,] pads)
    {
        var caseOptions = GetPassOptions();
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = NN.Pad(a, pads, PadMode.Constant, 0.0f);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldNopPad() }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

    [Theory]
    [MemberData(nameof(TestFoldNopPadNegativeData))]
    public void TestFoldNopPadNegative(int[] shape, int[,] pads)
    {
        var caseOptions = GetPassOptions();
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = NN.Pad(a, pads, PadMode.Constant, 0.0f);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldNopPad() }, caseOptions);

        Assert.Equal(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

    [Theory]
    [MemberData(nameof(TestFoldTwoPadsPositiveData))]
    public void TestFoldTwoPadsPositive(int[] shape, int[,] pads1, int[,] pads2)
    {
        var caseOptions = GetPassOptions();
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = NN.Pad(NN.Pad(a, pads1, PadMode.Constant, 0.0f), pads2, PadMode.Constant, 0.0f);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldTwoPads() }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

    [Theory]
    [MemberData(nameof(TestFoldTwoPadsNegativeData))]
    public void TestFoldTwoPadsNegative(int[] shape, int[,] pads1, float padValue1, int[,] pads2, float padValue2)
    {
        var caseOptions = GetPassOptions();
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = NN.Pad(NN.Pad(a, pads1, PadMode.Constant, padValue1), pads2, PadMode.Constant, padValue2);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldTwoPads() }, caseOptions);

        Assert.Equal(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
}
