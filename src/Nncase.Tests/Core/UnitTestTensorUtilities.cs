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
            new object[] { 23, new int[] { 24, 12, 4, 1 }, new int[] { 0, 1, 2, 3 }, 0 },
            new object[] { 23, new int[] { 24, 12, 4, 1 }, new int[] { 0, 1, 2, 3 }, 1 },
            new object[] { 11, new int[] { 24, 12, 4, 1 }, new int[] { 0, 1, 2, 3 }, 2 },
            new object[] { 3, new int[] { 24, 12, 4, 1 }, new int[] { 0, 1, 2, 3 }, 3 },
        };

    public static unsafe IEnumerable<object[]> TestGetIndexOverload2Data =>
        new[]
        {
            new object[] { 23, new Expr[] { 24, 12, 4, 1 }, new Expr[] { 0, 1, 2, 3 }, 0 },
            new object[] { 23, new Expr[] { 24, 12, 4, 1 }, new Expr[] { 0, 1, 2, 3 }, 1 },
            new object[] { 11, new Expr[] { 24, 12, 4, 1 }, new Expr[] { 0, 1, 2, 3 }, 2 },
            new object[] { 3, new Expr[] { 24, 12, 4, 1 }, new Expr[] { 0, 1, 2, 3 }, 3 },
        };

    public static unsafe IEnumerable<object[]> TestSplitStridesData =>
        new[]
        {
            new object[] { new int[] { 24, 12, 4, 1 }, Array.Empty<int>(), new int[] { 24, 12, 4, 1 }, Array.Empty<int>(), new int[4], 0, Array.Empty<int>(), 0 },
            new object[] { new int[] { 12, 4, 1 }, new int[] { 24 }, new int[] { 24, 12, 4, 1 }, new int[] { 0 }, new int[3], 0, new int[1], 0 },
            new object[] { new int[] { 24, 4, 1 }, new int[] { 12 }, new int[] { 24, 12, 4, 1 }, new int[] { 1 }, new int[3], 0, new int[1], 0 },
            new object[] { new int[] { 24, 12, 1 }, new int[] { 4 }, new int[] { 24, 12, 4, 1 }, new int[] { 2 }, new int[3], 0, new int[1], 0 },
            new object[] { new int[] { 24, 12, 4 }, new int[] { 1 }, new int[] { 24, 12, 4, 1 }, new int[] { 3 }, new int[3], 0, new int[1], 0 },
            new object[] { new int[] { 4, 1, }, new int[] { 24, 12 }, new int[] { 24, 12, 4, 1 }, new int[] { 0, 1 }, new int[2], 0, new int[2], 0 },
            new object[] { new int[] { 12, 1 }, new int[] { 24, 4 }, new int[] { 24, 12, 4, 1 }, new int[] { 0, 2 }, new int[2], 0, new int[2], 0 },
            new object[] { Array.Empty<int>(), new int[] { 24, 12, 4, 1 }, new int[] { 24, 12, 4, 1 }, new int[] { 0, 1, 2, 3 }, Array.Empty<int>(), 0, new int[4], 0 },
        };

    public static unsafe IEnumerable<object[]> TestTransformIndexByStridesData =>
        new[]
        {
            new object[] { 4, 20, new int[] { 24, 12, 4, 1 }, false, new int[] { 6, 2, 1, 1 } },
            new object[] { 5, 20, new int[] { 1, 4, 12, 24 }, true, new int[] { 6, 2, 1, 1 } },
        };

    [Fact]
    public void TestIsContiguousSlice()
    {
        var dim1 = new[] { 1, 512, 14, 14 };

        Assert.True(TensorUtilities.IsContiguousSlice(
          dim1,
          new[] { 0..1, 0..512, 0..14, 0..14 }));

        Assert.True(TensorUtilities.IsContiguousSlice(
          dim1,
          new[] { 0..1, 0..1, 0..1, 0..14 }));

        Assert.True(TensorUtilities.IsContiguousSlice(
          dim1,
          new[] { 0..1, 0..1, 0..1, 7..14 }));

        Assert.True(TensorUtilities.IsContiguousSlice(
          dim1,
          new[] { 0..1, 0..1, 7..14, 0..14 }));

        Assert.False(TensorUtilities.IsContiguousSlice(
          dim1,
          new[] { 0..1, 0..512, 0..7, 0..14 }));

        Assert.False(TensorUtilities.IsContiguousSlice(
            dim1,
            new[] { 0..1, 0..512, 0..7, 0..14, 0..1 }));

        Assert.False(TensorUtilities.IsContiguousSlice(
          dim1,
          new[] { 0..1, 10..512, 0..1, 0..1 }));

        Assert.False(TensorUtilities.IsContiguousSlice(
          dim1,
          new[] { 0..1, 0..512, 0..7, 0..1 }));

        var dim2 = new[] { 1, 512, 1, 196 };

        Assert.True(TensorUtilities.IsContiguousSlice(
              dim2,
              new[] { 0..1, 0..128, 0..1, 0..196 }));

        Assert.True(TensorUtilities.IsContiguousSlice(
              dim2,
              new[] { 0..1, 0..1, 0..1, 10..15 }));
    }

    // long GetProduct(ReadOnlySpan<int> dimensions, int startIndex = 0)
    [Fact]
    public void TestGetProductOverload1()
    {
        var a = new int[] { 1, 3, 16, 16 };
        Assert.Equal(a.Aggregate(1L, (x, y) => x * y), TensorUtilities.GetProduct(a));

        var b = new int[] { 1, -1, 16, 16 };
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorUtilities.GetProduct(b));
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
        var a = Enumerable.Range(1, 100).ToArray();
        Assert.True(TensorUtilities.IsAscending(a));

        var b = Enumerable.Repeat(1, 100).ToArray();
        Assert.True(TensorUtilities.IsAscending(b));

        var c = Enumerable.Range(1, 100).Reverse().ToArray();
        Assert.False(TensorUtilities.IsAscending(c));
    }

    [Fact]
    public void TestIsDescending()
    {
        var a = Enumerable.Range(1, 100).ToArray();
        Assert.False(TensorUtilities.IsDescending(a));

        var b = Enumerable.Repeat(1, 100).ToArray();
        Assert.True(TensorUtilities.IsDescending(b));

        var c = Enumerable.Range(1, 100).Reverse().ToArray();
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
        var shapes = new[] { 1, 2, 4, 8 };
        var strides = new[] { 1, 1, 1, 1 };
        var elementSize = 1;
        var getSize = TensorUtilities.GetSize(shapes, strides, elementSize);
        var result = 1;
        for (int i = 0; i < shapes.Length; i++)
        {
            result += (shapes[i] - 1) * strides[i];
        }

        Assert.Equal(result, getSize);
    }

    [Theory]
    [MemberData(nameof(TestSplitStridesData))]
    public void TestSplitStrides(int[] expectNewStrides, int[] expectSplitStrides, int[] strides, int[] splitAxes, int[] newStrides, int stridesOffset, int[] splitStrides, int splitStridesOffset)
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
        Assert.Equal(0, TensorUtilities.GetIndex(stride1, new int[] { 0 }));

        // exception
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorUtilities.GetIndex(stride1, new int[] { 0, 1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorUtilities.GetIndex(stride1, new int[] { 1 }));
    }

    [Theory]
    [MemberData(nameof(TestGetIndexOverload1Data))]
    public void TestGetIndexOverload1(int expect, int[] strides, int[] indices, int startFromDimension)
    {
        Assert.Equal(expect, TensorUtilities.GetIndex(strides, indices, startFromDimension));
    }

    // IR.Expr GetIndex(ReadOnlySpan<IR.Expr> strides, ReadOnlySpan<IR.Expr> indices, int startFromDimension = 0)
    [Fact]
    public void TestGetIndexOverload2Exception()
    {
        // stride is empty
        var stride1 = Array.Empty<Expr>();
        var indices = new Expr[] { 0 };
        var actual1 = TensorUtilities.GetIndex(stride1, indices);
        Assert.Equal(0, actual1.Evaluate().AsTensor().ToScalar<int>());

        // exception
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorUtilities.GetIndex(stride1, new Expr[] { 0, 1 }));
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
    public void TestTransformIndexByStrides(int expect, int index, int[] sourceStrides, bool sourceReverseStride, int[] transformStrides)
    {
        var actual = TensorUtilities.TransformIndexByStrides(index, sourceStrides, sourceReverseStride, transformStrides);
        Assert.Equal(expect, actual);
    }

    [Fact]
    public void TestIsContiguous()
    {
        Assert.True(TensorUtilities.IsContiguous(new int[] { 1, 2, 3, 4 }, new int[] { 24, 12, 4, 1 }));
        Assert.False(TensorUtilities.IsContiguous(new int[] { 1, 2, 3, 4 }, new int[] { 24, 12, 4 }));
        Assert.False(TensorUtilities.IsContiguous(new int[] { 1, 2, 3, 4 }, new int[] { 24, 12, 3, 1 }));
    }
}
