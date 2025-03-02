// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase;
using Nncase.IR;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestTensorUtilities
{
    public static unsafe IEnumerable<object[]> TestGetIndexOverload1Data =>
        new[]
        {
            new object[] { 23, new long[] { 24, 12, 4, 1 }, new long[] { 0, 1, 2, 3 }, 0 },
            [23, new long[] { 24, 12, 4, 1 }, new long[] { 0, 1, 2, 3 }, 1],
            [11, new long[] { 24, 12, 4, 1 }, new long[] { 0, 1, 2, 3 }, 2],
            [3, new long[] { 24, 12, 4, 1 }, new long[] { 0, 1, 2, 3 }, 3],
        };

    public static unsafe IEnumerable<object[]> TestGetIndexOverload2Data =>
        new[]
        {
            new object[] { 23, new Expr[] { 24L, 12L, 4L, 1L }, new Expr[] { 0L, 1L, 2L, 3L }, 0 },
            [23, new Expr[] { 24L, 12L, 4L, 1L }, new Expr[] { 0L, 1L, 2L, 3L }, 1],
            [11, new Expr[] { 24L, 12L, 4L, 1L }, new Expr[] { 0L, 1L, 2L, 3L }, 2],
            [3, new Expr[] { 24L, 12L, 4L, 1L }, new Expr[] { 0L, 1L, 2L, 3L }, 3],
        };

    public static unsafe IEnumerable<object[]> TestSplitStridesData =>
        new[]
        {
            new object[] { new long[] { 24, 12, 4, 1 }, Array.Empty<long>(), new long[] { 24, 12, 4, 1 }, Array.Empty<int>(), new long[4], 0, Array.Empty<long>(), 0L },
            [new long[] { 12, 4, 1 }, new long[] { 24 }, new long[] { 24, 12, 4, 1 }, new int[] { 0 }, new long[3], 0, new long[1], 0L],
            [new long[] { 24, 4, 1 }, new long[] { 12 }, new long[] { 24, 12, 4, 1 }, new int[] { 1 }, new long[3], 0, new long[1], 0L],
            [new long[] { 24, 12, 1 }, new long[] { 4 }, new long[] { 24, 12, 4, 1 }, new int[] { 2 }, new long[3], 0, new long[1], 0L],
            [new long[] { 24, 12, 4 }, new long[] { 1 }, new long[] { 24, 12, 4, 1 }, new int[] { 3 }, new long[3], 0, new long[1], 0L],
            [new long[] { 4, 1, }, new long[] { 24, 12 }, new long[] { 24, 12, 4, 1 }, new int[] { 0, 1 }, new long[2], 0, new long[2], 0L],
            [new long[] { 12, 1 }, new long[] { 24, 4 }, new long[] { 24, 12, 4, 1 }, new int[] { 0, 2 }, new long[2], 0, new long[2], 0L],
            [Array.Empty<long>(), new long[] { 24, 12, 4, 1 }, new long[] { 24, 12, 4, 1 }, new int[] { 0, 1, 2, 3 }, Array.Empty<long>(), 0, new long[4], 0L],
        };

    public static unsafe IEnumerable<object[]> TestTransformIndexByStridesData =>
        new[]
        {
            new object[] { 4, 20, new long[] { 24, 12, 4, 1 }, false, new long[] { 6, 2, 1, 1 } },
            [5, 20, new long[] { 1, 4, 12, 24 }, true, new long[] { 6, 2, 1, 1 }],
        };

    [Fact]
    public void TestIsContiguousSlice()
    {
        var dim1 = new long[] { 1, 512, 14, 14 };
        int start;
        Assert.True(TensorUtilities.IsContiguousSlice(
          dim1,
          [0..1, 0..512, 0..14, 0..14],
          out start));
        Assert.Equal(0, start);

        Assert.True(TensorUtilities.IsContiguousSlice(
          dim1,
          [0..1, 0..1, 0..1, 0..14],
          out start));
        Assert.Equal(0, start);

        Assert.True(TensorUtilities.IsContiguousSlice(
          dim1,
          [0..1, 0..1, 0..1, 7..14],
          out start));
        Assert.Equal(0, start);

        Assert.True(TensorUtilities.IsContiguousSlice(
          dim1,
          [0..1, 0..1, 7..14, 0..14],
          out start));
        Assert.Equal(0, start);

        Assert.False(TensorUtilities.IsContiguousSlice(
          dim1,
          [0..1, 0..512, 0..7, 0..14],
          out start));
        Assert.Equal(2, start);

        Assert.False(TensorUtilities.IsContiguousSlice(
            dim1,
            [0..1, 0..512, 0..7, 0..14, 0..1],
            out start));
        Assert.Equal(4, start);

        Assert.False(TensorUtilities.IsContiguousSlice(
          dim1,
          [0..1, 10..512, 0..1, 0..1],
          out start));
        Assert.Equal(2, start);

        Assert.False(TensorUtilities.IsContiguousSlice(
          dim1,
          [0..1, 0..512, 0..7, 0..1],
          out start));
        Assert.Equal(3, start);

        var dim2 = new long[] { 1, 512, 1, 196 };

        Assert.True(TensorUtilities.IsContiguousSlice(
              dim2,
              [0..1, 0..128, 0..1, 0..196],
              out start));
        Assert.Equal(0, start);

        Assert.True(TensorUtilities.IsContiguousSlice(
              dim2,
              [0..1, 0..1, 0..1, 10..15],
              out start));
        Assert.Equal(0, start);
    }

    // long GetProduct(ReadOnlySpan<int> dimensions, int startIndex = 0)
    [Fact]
    public void TestGetProductOverload1()
    {
        var a = new int[] { 1, 3, 16, 16 };
        Assert.Equal(a.Aggregate(1L, (x, y) => x * y), TensorUtilities.GetProduct(a));

        var b = new int[] { 1, -1, 16, 16 };
        Assert.Equal(b.Aggregate(1L, (x, y) => x * y), TensorUtilities.GetProduct(b));
    }

    // IR.Expr GetProduct(IEnumerable<IR.Expr> dimensions, int startIndex = 0)
    [Fact]
    public void TestGetProductOverload2()
    {
        var dims1 = Array.Empty<Expr>();
        var actual1 = TensorUtilities.GetProduct(dims1);
        Assert.Equal(1, actual1.Evaluate().AsTensor().ToScalar<long>());

        var dims2 = new Expr[] { 1, 2, 3, 4 };
        var expect2 = dims2.Select(x => x.Evaluate().AsTensor().ToScalar<int>()).ToArray<int>().Aggregate(1L, (x, y) => x * y);
        var actual2 = TensorUtilities.GetProduct(dims2);
        Assert.True(CompilerServices.InferenceType(actual2));
        Assert.Equal(expect2, actual2.Evaluate().AsTensor().ToScalar<long>());
    }

    [Fact]
    public void TestIsAsending()
    {
        var a = Enumerable.Range(1, 100).ToArray().ToLongs();
        Assert.True(TensorUtilities.IsAscending(a));

        var b = Enumerable.Repeat(1, 100).ToArray().ToLongs();
        Assert.True(TensorUtilities.IsAscending(b));

        var c = Enumerable.Range(1, 100).Reverse().ToArray().ToLongs();
        Assert.False(TensorUtilities.IsAscending(c));
    }

    [Fact]
    public void TestIsDescending()
    {
        var a = Enumerable.Range(1, 100).ToArray().ToLongs();
        Assert.False(TensorUtilities.IsDescending(a));

        var b = Enumerable.Repeat(1, 100).ToArray().ToLongs();
        Assert.True(TensorUtilities.IsDescending(b));

        var c = Enumerable.Range(1, 100).Reverse().ToArray().ToLongs();
        Assert.True(TensorUtilities.IsDescending(c));
    }

    // int[] GetStrides(ReadOnlySpan<int> dimensions, bool reverseStride = false)
    [Fact]
    public void TestGetStridesOverload1()
    {
        // dimensions.IsEmpty
        var dims1 = Array.Empty<int>();
        Assert.True(TensorUtilities.GetStrides(dims1) == Array.Empty<int>());

        // reverseStride == false
        var dims2 = new int[] { 1, 2, 3, 4 };
        var expect1 = new int[] { 24, 12, 4, 1 };
        var actual1 = TensorUtilities.GetStrides(dims2);
        Assert.Equal(expect1, actual1);

        // reverseStride == true
        var expect2 = new int[] { 1, 1, 2, 6 };
        var actual2 = TensorUtilities.GetStrides(dims2, true);
        Assert.Equal(expect2, actual2);
    }

    // IEnumerable<IR.Expr> GetStrides(IEnumerable<IR.Expr> dimensions, bool reverseStride = false)
    [Fact]
    public void TestGetStridesOverload2()
    {
        var a = new Expr[] { 1, 2, 3, 4 };

        // reverseStride == false
        var expect1 = new int[] { 24, 12, 4, 1 };
        var actual1 = TensorUtilities.GetStrides(a).Select(x => x.Evaluate().AsTensor().ToScalar<int>()).ToArray<int>();
        Assert.Equal(expect1, actual1);

        // reverseStride == true
        var expect2 = new int[] { 1, 1, 2, 6 };
        var actual2 = TensorUtilities.GetStrides(a, true).Select(x => x.Evaluate().AsTensor().ToScalar<int>()).ToArray<int>();
        Assert.Equal(expect2, actual2);

        var b = Array.Empty<Expr>();
        var expect3 = Array.Empty<int>();
        var actual3 = TensorUtilities.GetStrides(b, true).Select(x => x.Evaluate().AsTensor().ToScalar<int>()).ToArray<int>();
        Assert.Equal(expect3, actual3);
    }

    [Fact]
    public void TestGetSize()
    {
        var shapes = new long[] { 1, 2, 4, 8 };
        var strides = new long[] { 1, 1, 1, 1 };
        var elementSize = 1;
        var getSize = TensorUtilities.GetSize(shapes, strides, elementSize);
        long result = 1;
        for (int i = 0; i < shapes.Length; i++)
        {
            result += (shapes[i] - 1) * strides[i];
        }

        Assert.Equal(result, getSize);
    }

    [Theory]
    [MemberData(nameof(TestSplitStridesData))]
    public void TestSplitStrides(long[] expectNewStrides, long[] expectSplitStrides, long[] strides, int[] splitAxes, long[] newStrides, long stridesOffset, long[] splitStrides, long splitStridesOffset)
    {
        TensorUtilities.SplitStrides(strides, splitAxes, newStrides, stridesOffset, splitStrides, splitStridesOffset);
        Assert.Equal(expectNewStrides, newStrides);
        Assert.Equal(expectSplitStrides, splitStrides);
    }

    // int GetIndex(ReadOnlySpan<int> strides, ReadOnlySpan<int> indices, int startFromDimension = 0)
    [Fact]
    public void TestGetIndexOverload1Exception()
    {
        // stride is empty
        var stride1 = Array.Empty<int>();
        Assert.Equal(0, TensorUtilities.GetIndex(stride1, []));

        // exception
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorUtilities.GetIndex(stride1, [0, 1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorUtilities.GetIndex(stride1, [1]));
    }

    [Theory]
    [MemberData(nameof(TestGetIndexOverload1Data))]
    public void TestGetIndexOverload1(long expect, long[] strides, long[] indices, int startFromDimension)
    {
        Assert.Equal(expect, TensorUtilities.GetIndex(strides, indices, startFromDimension));
    }

    // IR.Expr GetIndex(ReadOnlySpan<IR.Expr> strides, ReadOnlySpan<IR.Expr> indices, int startFromDimension = 0)
    [Fact]
    public void TestGetIndexOverload2Exception()
    {
        // stride is empty
        var stride1 = Array.Empty<Expr>();
        var indices = Array.Empty<Expr>();
        var actual1 = TensorUtilities.GetIndex(stride1, indices);
        Assert.Equal(0, actual1.Evaluate().AsTensor().ToScalar<int>());

        // exception
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorUtilities.GetIndex(stride1, [0, 1]));
    }

    [Theory]
    [MemberData(nameof(TestGetIndexOverload2Data))]
    public void TestGetIndexOverload2(Expr expect, Expr[] strides, Expr[] indices, int startFromDimension)
    {
        var expr = TensorUtilities.GetIndex(strides, indices, startFromDimension);
        var actual = expr.Evaluate().AsTensor().ToScalar<int>();
        Assert.Equal(expect, actual);
    }

    [Theory]
    [MemberData(nameof(TestTransformIndexByStridesData))]
    public void TestTransformIndexByStrides(int expect, int index, long[] sourceStrides, bool sourceReverseStride, long[] transformStrides)
    {
        var actual = TensorUtilities.TransformIndexByStrides(index, sourceStrides, sourceReverseStride, transformStrides);
        Assert.Equal(expect, actual);
    }

    [Fact]
    public void TestIsContiguous()
    {
        Assert.True(TensorUtilities.IsContiguous([1, 2, 3, 4], [24, 12, 4, 1]));
        Assert.False(TensorUtilities.IsContiguous([1, 2, 3, 4], [24, 12, 4]));
        Assert.False(TensorUtilities.IsContiguous([1, 2, 3, 4], [24, 12, 3, 1]));
    }
}
