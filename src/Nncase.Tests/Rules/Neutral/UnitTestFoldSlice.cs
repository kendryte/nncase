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
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public class UnitTestFoldSlice : TransformTestBase
{
    public static IEnumerable<object[]> TestFoldNopSlicePositiveData =>
        new[]
        {
            new object[] { new long[] { 2 }, new long[] { 0 }, new long[] { 2 }, new[] { 0 }, new[] { 1 } },
            new object[] { new long[] { 2, 4,  }, new long[] { 0 }, new long[] { 4 }, new[] { 1 }, new[] { 1 } },
            new object[] { new long[] { 2, 4,  }, new long[] { 0, 0 }, new long[] { 2, 4 }, new[] { 0, 1 }, new[] { 1, 1 } },
            new object[] { new long[] { 2, 4, 6, 8 }, new long[] { 0 }, new long[] { 6 }, new[] { 2 }, new[] { 1 } },
            new object[] { new long[] { 2, 4, 6, 8 }, new long[] { 0, 0 }, new long[] { 4, 6 }, new[] { 1, 2 }, new[] { 1, 1 } },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    public static IEnumerable<object[]> TestFoldNopSliceNegativeData =>
        new[]
        {
            new object[] { new long[] { 2 }, new long[] { 0 }, new long[] { 1 }, new[] { 0 }, new[] { 1 } },
            new object[] { new long[] { 2, 4,  }, new long[] { 0 }, new long[] { 2 }, new[] { 1 }, new[] { 2 } },
            new object[] { new long[] { 2, 4,  }, new long[] { 0, 2 }, new long[] { 2, 4 }, new[] { 0, 1 }, new[] { 1, 3 } },
            new object[] { new long[] { 2, 4, 6, 8 }, new long[] { 2 }, new long[] { 6 }, new[] { 2 }, new[] { -1 } },
            new object[] { new long[] { 2, 4, 6, 8 }, new long[] { 0, 1 }, new long[] { 4, 6 }, new[] { 1, 2 }, new[] { 1, 1 } },
            new object[] { new long[] { 2, 4, 6, 8 }, new long[] { 0, 0 }, new long[] { -1, 6 }, new[] { 1, 2 }, new[] { -1, -1 } },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    public static IEnumerable<object[]> TestFoldTwoSlicePositiveData =>
        new[]
        {
            new object[]
            {
                new long[] { 1, 4, 6, 8 },
                new long[] { 0 }, new long[] { 6 }, new[] { 3 }, new[] { 3 },
                new long[] { 0 }, new long[] { 4 }, new[] { 2 }, new[] { 2 },
            }, // Diff axis
            new object[]
            {
                new long[] { 4, 4, 6, 8 },
                new long[] { 0 }, new long[] { 6 }, new[] { 3 }, new[] { 3 },
                new long[] { 0 }, new long[] { 4 }, new[] { 2 }, new[] { 2 },
            }, // negative axis
            new object[]
            {
                new long[] { 3, 4, 6, 8 },
                new long[] { 0 }, new long[] { -1 }, new[] { 3 }, new[] { 3 },
                new long[] { -5 }, new long[] { 4 }, new[] { 2 }, new[] { 2 },
            }, // negative begin|end
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    public static TheoryData<long[], long[], long[], int[], int[], long[], long[], int[], int[], int> TestFoldTwoSlicePositiveData2 { get; } = new()
    {
      {
        new long[] { 1, 3, 640, 640 },
        new long[] { 0 },
        new long[] { int.MaxValue },
        new[] { 2 },
        new[] { 2 },
        new long[] { 0 },
        new long[] { int.MaxValue },
        new[] { 3 },
        new[] { 2 },
        0
      },
      {
        new long[] { 1, 3, 640, 640 },
        new long[] { 0 },
        new long[] { int.MaxValue },
        new[] { 2 },
        new[] { 2 },
        new long[] { 1 },
        new long[] { int.MaxValue },
        new[] { 3 },
        new[] { 2 },
        1
      },
      {
        new long[] { 1, 3, 640, 640 },
        new long[] { 1 },
        new long[] { int.MaxValue },
        new[] { 2 },
        new[] { 2 },
        new long[] { 0 },
        new long[] { int.MaxValue },
        new[] { 3 },
        new[] { 2 },
        2
      },
      {
        new long[] { 1, 3, 640, 640 },
        new long[] { 1 },
        new long[] { int.MaxValue },
        new[] { 2 },
        new[] { 2 },
        new long[] { 1 },
        new long[] { int.MaxValue },
        new[] { 3 },
        new[] { 2 },
        3
      },
    };

    [Theory]
    [MemberData(nameof(TestFoldNopSlicePositiveData))]
    public void TestFoldNopSlicePositive(long[] shape, long[] begins, long[] ends, int[] axes, int[] strides, int index)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Slice(a, begins, ends, axes, strides);
        TestMatched<FoldNopSlice>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestFoldNopSliceNegativeData))]
    public void TestFoldNopSliceNegative(long[] shape, long[] begins, long[] ends, int[] axes, int[] strides, int index)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Slice(a, begins, ends, axes, strides);
        TestNotMatch<FoldNopSlice>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestFoldTwoSlicePositiveData2))]
    [MemberData(nameof(TestFoldTwoSlicePositiveData))]
    public void TestFoldTwoSlicePositive(long[] shape, long[] begins1, long[] ends1, int[] axes1, int[] strides1, long[] begins2, long[] ends2, int[] axes2, int[] strides2, int index)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var first_slice = Tensors.Slice(a, begins1, ends1, axes1, strides1);
        var rootPre = Tensors.Slice(first_slice, begins2, ends2, axes2, strides2);
        TestMatched<FoldTwoSlices>(rootPre);
    }
}
