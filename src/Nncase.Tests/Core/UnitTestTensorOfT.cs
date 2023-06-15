// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestTensorOfT
{
    public static unsafe IEnumerable<object[]> TestSetValueData =>
        new[]
        {
            new object[] { 0, 0 },
            new object[] { 1, 100 },
            new object[] { 2, 200 },
            new object[] { 3, 300 },
            new object[] { 4, 400 },
            new object[] { 5, 500 },
            new object[] { 6, 600 },
            new object[] { 7, 700 },
        };

    [Fact]
    public void TestICollection()
    {
        var a1 = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a1);
        var t = new Tensor<int>(memory, new int[] { 1, 1, 2, 4 });
        ICollection<int> c = t;
        Assert.Equal(a1.Length, c.Count);
        Assert.False(c.IsReadOnly);

        Assert.Throws<InvalidOperationException>(() => c.Add(100));
        Assert.Throws<InvalidOperationException>(() => c.Remove(8));

        var a2 = new int[a1.Length];
        c.CopyTo(a2, 0);
        Assert.True(Enumerable.SequenceEqual(a1, a2));

        c.Clear();

        Assert.Equal(a1.Length, c.Count);
        for (int i = 0; i < a1.Length; i++)
        {
            Assert.Equal(0, t.GetValue(i));
        }
    }

    [Fact]
    public void TestIReadOnlyCollection()
    {
        var length = 100;
        var a = new float[length];
        var t = new Tensor<float>(length);
        var c = (IReadOnlyCollection<float>)t;
        Assert.Equal(length, c.Count);
        Assert.Equal(a, c.ToArray<float>());
    }

    [Fact]
    public void TestIReadOnlyList()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);
        var t = new Tensor<int>(memory, new int[] { 8 });
        var list = (IReadOnlyList<int>)t;
        for (int i = 0; i < a.Length; i++)
        {
            Assert.Equal(a[i], list[i]);
        }
    }

    [Fact]
    public void TestIList()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);
        var t = new Tensor<int>(memory, new int[] { 8 });
        var list = (IList<int>)t;

        for (int i = 0; i < a.Length; i++)
        {
            list[i] = i;
        }

        for (int i = 0; i < a.Length; i++)
        {
            Assert.Equal(i, list[i]);
        }

        Assert.Equal(1, list.IndexOf(1));
        Assert.Throws<InvalidOperationException>(() => list.Insert(0, 100));
        Assert.Throws<InvalidOperationException>(() => list.RemoveAt(0));
    }

    [Fact]
    public void TestImplicitOperator()
    {
        var scalar = 1F;
        var a = (Tensor<float>)scalar;
        Assert.Equal(scalar, a[0]);
    }

    [Fact]
    public void TestClone()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);
        var t1 = new Tensor<int>(memory, new int[] { 8 });
        var t2 = t1.Clone();
        Assert.Equal(t1, t2);
        Assert.NotSame(t1, t2);
    }

    [Fact]
    public void TestCloneEmpty()
    {
        var t1 = new Tensor<float>(new int[] { 1, 2, 3, 4 });
        var t2 = t1.CloneEmpty<int>(new int[] { 1, 3, 16, 16 });
        var a = (ICollection<int>)t2;
        Assert.Equal(1 * 3 * 16 * 16, a.Count);
    }

    [Fact]
    public void TestGetValue()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);
        var t = new Tensor<int>(memory, new int[] { 2, 4 });
        Assert.Equal(2, t.GetValue(1));
        Assert.Equal(7, t.GetValue(6));
        Assert.Throws<IndexOutOfRangeException>(() => t.GetValue(a.Length));
    }

    [Fact]
    public void TestReshape()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);

        var t1 = new Tensor<int>(memory, new int[] { 1, 1, 2, 4 });
        Assert.Throws<ArgumentException>(() => t1.Reshape(new int[] { 1, 1, 4, 3 }));

        var t2 = t1.Reshape(new int[] { 1, 1, 4, 2 });
        Assert.True(t2.Buffer.Span.SequenceEqual(t1.Buffer.Span));
    }

    [Theory]
    [MemberData(nameof(TestSetValueData))]
    public void TestSetValue(int index, int value)
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);
        var t = new Tensor<int>(memory, new int[] { 2, 4 });
        t.SetValue(index, value);
        Assert.Equal(value, t.GetValue(index));
        Assert.Throws<IndexOutOfRangeException>(() => t.SetValue(a.Length, 9));
    }

    [Fact]
    public void TestFill()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);
        var t = new Tensor<int>(memory, new int[] { 2, 4 });
        int value = 100;
        t.Fill(value);
        for (int i = 0; i < a.Length; i++)
        {
            Assert.Equal(value, t.GetValue(i));
        }
    }

    [Fact]
    public void TestContains()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);
        var t = new Tensor<int>(memory, new int[] { 2, 4 });
        Assert.Contains(1, t);
        Assert.Contains(8, t);
    }

    [Fact]
    public void TestTryGetIndicesOf()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);
        var t = new Tensor<int>(memory, new int[] { 1, 1, 2, 4 });

        var indices1 = new int[] { 1, 1, 1, 1 };
        Assert.True(t.TryGetIndicesOf(7, indices1));
        Assert.Equal(0, indices1[0]);
        Assert.Equal(0, indices1[1]);
        Assert.Equal(1, indices1[2]);
        Assert.Equal(2, indices1[3]);

        var indices2 = new int[] { 1, 1 };
        Assert.Throws<ArgumentException>(() => t.TryGetIndicesOf(7, indices2));

        var indices3 = new int[] { 1, 1, 1, 1 };
        Assert.False(t.TryGetIndicesOf(100, indices3));
    }

    // bool Equals(object? obj)
    [Fact]
    public void TestEqualsOverload1()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);

        var t1 = new Tensor<int>(memory, new int[] { 1, 1, 2, 4 });
        var t2 = new Tensor<int>(memory, new int[] { 2, 4 });
        Assert.False(t1.Equals((object)t2));

        var t3 = new Tensor<int>(memory, new int[] { 1, 1, 4, 2 });
        Assert.False(t1.Equals((object)t3));

        var t4 = new Tensor<int>(memory, new int[] { 1, 1, 2, 4 });
        Assert.True(t1.Equals((object)t4));

        Assert.False(t1.Equals(a));
    }

    // bool Equals(Tensor<T>? other)
    [Fact]
    public void TestEqualsOverload2()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);

        var t1 = new Tensor<int>(memory, new int[] { 1, 1, 2, 4 });
        Tensor<int>? n = null;
        Assert.NotStrictEqual(t1, n);
        Assert.False(t1.Equals(n));

        var t2 = new Tensor<int>(memory, new int[] { 2, 4 });
        Assert.NotStrictEqual(t1, t2);

        var t3 = new Tensor<int>(memory, new int[] { 1, 1, 4, 2 });
        Assert.NotStrictEqual(t1, t3);

        var t4 = new Tensor<int>(memory, new int[] { 1, 1, 2, 4 });
        Assert.StrictEqual(t1, t4);
    }
}
