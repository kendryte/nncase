// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using Nncase;
using Nncase.IR;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestShape
{
    [Fact]
    public void TestIEnumerableDimension()
    {
        var a = new Dimension[] { 1, 3, 2, 2 };
        var s = new RankedShape((IEnumerable<Dimension>)a);
        Assert.Equal(ShapeKind.Fixed, s.Kind);
        Assert.True(s.IsFixed);
        Assert.False(s.IsInvalid);
        Assert.False(s.IsUnranked);
        Assert.False(s.HasUnknownDimension);
        Assert.True(s.IsRanked);
        Assert.False(s.IsScalar);
        Assert.Equal(a.Length, s.Rank);
        Assert.Equal(a.Length, s.Count);
        Assert.Equal(12, s.Size);
    }

    [Fact]
    public void TestIEnumerableLong()
    {
        var a = new long[] { 1, 3, 2, 2 };
        var s = new RankedShape(a);
        Assert.Equal(ShapeKind.Fixed, s.Kind);
        Assert.True(s.IsFixed);
        Assert.False(s.IsInvalid);
        Assert.False(s.IsUnranked);
        Assert.False(s.HasUnknownDimension);
        Assert.True(s.IsRanked);
        Assert.False(s.IsScalar);
        Assert.Equal(a.Length, s.Rank);
        Assert.Equal(a.Length, s.Count);
        Assert.Equal(12, s.Size);
    }

    [Fact]
    public void TestIEnumerableInt()
    {
        var a = new int[] { 1, 3, 2, 2 };
        var s = new RankedShape((IEnumerable<int>)a);
        Assert.Equal(ShapeKind.Fixed, s.Kind);
        Assert.True(s.IsFixed);
        Assert.False(s.IsInvalid);
        Assert.False(s.IsUnranked);
        Assert.False(s.HasUnknownDimension);
        Assert.True(s.IsRanked);
        Assert.False(s.IsScalar);
        Assert.Equal(a.Length, s.Rank);
        Assert.Equal(a.Length, s.Count);
        Assert.Equal(12, s.Size);
    }

    [Fact]
    public void TestIntArray()
    {
        var a = new int[] { 1, 3, 2, 2 };
        var s = new RankedShape(a);
        Assert.Equal(ShapeKind.Fixed, s.Kind);
        Assert.True(s.IsFixed);
        Assert.False(s.IsInvalid);
        Assert.False(s.IsUnranked);
        Assert.False(s.HasUnknownDimension);
        Assert.True(s.IsRanked);
        Assert.False(s.IsScalar);
        Assert.Equal(a.Length, s.Rank);
        Assert.Equal(a.Length, s.Count);
        Assert.Equal(12, s.Size);
    }

    [Fact]
    public void TestImplicitOeratorIntArray()
    {
        var a = new int[] { 1, 3, 2, 2 };
        RankedShape s = a;
        Assert.Equal(ShapeKind.Fixed, s.Kind);
        Assert.True(s.IsFixed);
        Assert.False(s.IsInvalid);
        Assert.False(s.IsUnranked);
        Assert.False(s.HasUnknownDimension);
        Assert.True(s.IsRanked);
        Assert.False(s.IsScalar);
        Assert.Equal(a.Length, s.Rank);
        Assert.Equal(a.Length, s.Count);
        Assert.Equal(12, s.Size);
    }

    [Fact]
    public void TestDimensionArray()
    {
        var a = new Dimension[] { 1, 3, 2, 2 };
        var s = new RankedShape(a);
        Assert.Equal(ShapeKind.Fixed, s.Kind);
        Assert.True(s.IsFixed);
        Assert.False(s.IsInvalid);
        Assert.False(s.IsUnranked);
        Assert.False(s.HasUnknownDimension);
        Assert.True(s.IsRanked);
        Assert.False(s.IsScalar);
        Assert.Equal(a.Length, s.Rank);
        Assert.Equal(a.Length, s.Count);
        Assert.Equal(12, s.Size);
    }

    [Fact]
    public void TestImplicitOeratorDimensionArray()
    {
        var a = new Dimension[] { 1, 3, 2, 2 };
        RankedShape s = a;
        Assert.Equal(ShapeKind.Fixed, s.Kind);
        Assert.True(s.IsFixed);
        Assert.False(s.IsInvalid);
        Assert.False(s.IsUnranked);
        Assert.False(s.HasUnknownDimension);
        Assert.True(s.IsRanked);
        Assert.False(s.IsScalar);
        Assert.Equal(a.Length, s.Rank);
        Assert.Equal(a.Length, s.Count);
        Assert.Equal(12, s.Size);
    }

    [Fact]
    public void TestInvalid()
    {
        var s = Shape.Invalid;
        Assert.False(s.IsFixed);
        Assert.True(s.IsInvalid);
        Assert.False(s.IsUnranked);
        Assert.False(s.HasUnknownDimension);
        Assert.False(s.IsRanked);
        Assert.False(s.IsScalar);
        Assert.Throws<InvalidOperationException>(() => s.Rank);
        Assert.Throws<InvalidOperationException>(s.GetEnumerator);
        Assert.Throws<InvalidOperationException>(s.ToValueArray);
        Assert.Throws<InvalidOperationException>(s.ToValueArrayExpr);
    }

    [Fact]
    public void TestUnranked()
    {
        var s = Shape.Unranked;
        Assert.False(s.IsFixed);
        Assert.False(s.IsInvalid);
        Assert.True(s.IsUnranked);
        Assert.False(s.HasUnknownDimension);
        Assert.False(s.IsRanked);
        Assert.False(s.IsScalar);
        Assert.Throws<InvalidOperationException>(() => s.Rank);
        Assert.Throws<InvalidOperationException>(s.GetEnumerator);
        Assert.Throws<InvalidOperationException>(s.ToValueArray);
    }

    [Fact]
    public void TestScalar()
    {
        var s = Shape.Scalar;
        Assert.True(s.IsFixed);
        Assert.False(s.IsInvalid);
        Assert.False(s.IsUnranked);
        Assert.False(s.HasUnknownDimension);
        Assert.True(s.IsRanked);
        Assert.True(s.IsScalar);
        Assert.Equal(0, s.Rank);
        Assert.Empty(s);
        Assert.Equal(1, s.Size);
        Assert.Empty(s.ToValueArray());
    }

    [Fact]
    public void TestScalar2()
    {
        var s = new RankedShape(Array.Empty<Dimension>());
        Assert.True(s.IsFixed);
        Assert.False(s.IsInvalid);
        Assert.False(s.IsUnranked);
        Assert.False(s.HasUnknownDimension);
        Assert.True(s.IsRanked);
        Assert.True(s.IsScalar);
        Assert.Equal(0, s.Rank);
        Assert.Empty(s);
        Assert.Equal(1, s.Size);
        Assert.Empty(s.ToValueArray());
    }

    [Fact]
    public void TestEqual()
    {
        var a1 = new int[] { 1, 3, 2, 2 };
        var a2 = new int[] { 1, 3, 2, 2 };
        var a3 = new int[] { 1, 3, 4, 4 };

        var s1 = new RankedShape(a1);
        var s2 = new RankedShape(a2);
        var s3 = new RankedShape(a3);

        Assert.True(s1 == s2);
        Assert.False(s1 == s3);
    }

    [Fact]
    public void TestNotEqual()
    {
        var a1 = new int[] { 1, 3, 2, 2 };
        var a2 = new int[] { 1, 3, 2, 2 };
        var a3 = new int[] { 1, 3, 4, 4 };

        var s1 = new RankedShape(a1);
        var s2 = new RankedShape(a2);
        var s3 = new RankedShape(a3);

        Assert.False(s1 != s2);
        Assert.True(s1 != s3);
    }

    [Fact]
    public void TestEquals()
    {
        var a1 = new int[] { 1, 3, 2, 2 };
        var a2 = new int[] { 1, 3, 2, 2 };
        var a3 = new int[] { 1, 3, 4, 4 };

        var s1 = new RankedShape(a1);
        var s2 = new RankedShape(a2);
        var s3 = new RankedShape(a3);

        Assert.Equal(s1, s2);
        Assert.NotEqual(s1, s3);

        Assert.Equal(s1, (object)s2);
    }

    [Fact]
    public void TestInsertAndCloneOverload1()
    {
        int index = 1;
        int item = 3;
        var a = new int[] { 1, 2, 2 };
        List<int> list = new();
        list.AddRange(a);
        list.Insert(index, item);
        var expected = new RankedShape(list);
        var actual = new RankedShape(a).InsertAndClone(index, item);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TestInsertAndCloneOverload2()
    {
        int index = 1;
        var items = new int[] { 3, 2 };
        var dimensions = new Dimension[] { 3, 2 };
        var a = new int[] { 1, 2 };
        List<int> list = new();
        list.AddRange(a);
        list.InsertRange(index, items);
        var expected = new RankedShape(list);
        var actual = new RankedShape(a).InsertAndClone(index, dimensions);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TestToValueArray()
    {
        int index = 1;
        var items = new long[] { 3, 2 };
        var dimensions = new Dimension[] { 3, 2 };
        var a = new long[] { 1, 2 };
        List<long> list = new();
        list.AddRange(a);
        list.InsertRange(index, items);
        var expected = list.ToArray();

        var s = new RankedShape(a).InsertAndClone(index, dimensions);
        var actual = s.ToValueArray();
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TestToString()
    {
        Shape s = Shape.Invalid;
        Assert.Equal("[invalid]", s.ToString());

        s = Shape.Unranked;
        Assert.Equal("[*]", s.ToString());

        var a = new int[] { 1, 3, 2, 2 };
        s = a;
        Assert.Equal("[1,3,2,2]", s.ToString());
    }
}
