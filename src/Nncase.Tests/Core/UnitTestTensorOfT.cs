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
            new object[] { 0, 100 },
            new object[] { 1, 200 },
            new object[] { 3, 300 },
            new object[] { 4, 400 },
            new object[] { 5, 500 },
            new object[] { 6, 600 },
            new object[] { 7, 700 },
        };

    [Fact]
    public void TestICollection()
    {
        var array1 = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var momory = new Memory<int>(array1);
        var t = new Tensor<int>(momory, new int[] { 1, 1, 2, 4 });
        ICollection<int> c = t;
        Assert.Equal(array1.Length, c.Count);
        Assert.False(c.IsReadOnly);

        Assert.Throws<InvalidOperationException>(() => c.Add(100));
        Assert.Throws<InvalidOperationException>(() => c.Remove(8));

        var array2 = new int[array1.Length];
        c.CopyTo(array2, 0);
        Assert.True(Enumerable.SequenceEqual(array1, array2));

        c.Clear();

        Assert.Equal(array1.Length, c.Count);
        for (int i = 0; i < array1.Length; i++)
        {
            Assert.Equal(0, t.GetValue(i));
        }
    }

    [Fact]
    public void TestIReadOnlyCollection()
    {
        var length = 100;
        var tensor = new Tensor<float>(length);
        var a = (IReadOnlyCollection<float>)tensor;
        Assert.Equal(length, a.Count);
    }

    [Fact]
    public void TestIReadOnlyList()
    {
        var array = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var momory = new Memory<int>(array);
        var tensor = new Tensor<int>(momory, new int[] { 8 });
        var list = (IReadOnlyList<int>)tensor;
        for (int i = 0; i < array.Length; i++)
        {
            Assert.Equal(array[i], list[i]);
        }
    }

    [Fact]
    public void TestIList()
    {
        var array = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var momory = new Memory<int>(array);
        var tensor = new Tensor<int>(momory, new int[] { 8 });
        var list = (IList<int>)tensor;

        for (int i = 0; i < array.Length; i++)
        {
            list[i] = i;
        }

        for (int i = 0; i < array.Length; i++)
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
        var array = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var momory = new Memory<int>(array);
        var t1 = new Tensor<int>(momory, new int[] { 8 });
        var t2 = t1.Clone();
        for (int i = 0; i < array.Length; i++)
        {
            Assert.Equal(t1[i], t2[i]);
        }
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
        var array = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var momory = new Memory<int>(array);
        var t = new Tensor<int>(momory, new int[] { 2, 4 });
        Assert.Equal(2, t.GetValue(1));
        Assert.Equal(7, t.GetValue(6));
    }

    [Fact]
    public void TestReshape()
    {
        var array = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var momory = new Memory<int>(array);

        var t1 = new Tensor<int>(momory, new int[] { 1, 1, 2, 4 });
        Assert.Throws<ArgumentException>(() => t1.Reshape(new int[] { 1, 1, 4, 3 }));

        var t2 = t1.Reshape(new int[] { 1, 1, 4, 2 });
        Assert.True(t2.Buffer.Span.SequenceEqual(t1.Buffer.Span));
    }

    [Theory]
    [MemberData(nameof(TestSetValueData))]
    public void TestSetValue(int index, int value)
    {
        var array = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var momory = new Memory<int>(array);
        var t = new Tensor<int>(momory, new int[] { 2, 4 });
        t.SetValue(index, value);
        Assert.Equal(value, t.GetValue(index));
    }

    [Fact]
    public void TestFill()
    {
        var array = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var momory = new Memory<int>(array);
        var t = new Tensor<int>(momory, new int[] { 2, 4 });
        int value = 100;
        t.Fill(value);
        for (int i = 0; i < array.Length; i++)
        {
            Assert.Equal(value, t.GetValue(i));
        }
    }

    [Fact]
    public void TestContains()
    {
        var array = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var momory = new Memory<int>(array);
        var t = new Tensor<int>(momory, new int[] { 2, 4 });
        Assert.Contains(1, t);
        Assert.Contains(8, t);
    }

    [Fact]
    public void TestTryGetIndicesOf()
    {
        var array = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var momory = new Memory<int>(array);
        var t = new Tensor<int>(momory, new int[] { 1, 1, 2, 4 });

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

    [Fact]
    public void TestEquals1()
    {
        var array = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var momory = new Memory<int>(array);

        var t1 = new Tensor<int>(momory, new int[] { 1, 1, 2, 4 });
        Tensor<int> n = null;

        // Assert.NotEqual(t1, n);
        Assert.False(t1.Equals(n));

        var t2 = new Tensor<int>(momory, new int[] { 2, 4 });
        Assert.NotEqual(t1, t2);

        var t3 = new Tensor<int>(momory, new int[] { 1, 1, 4, 2 });
        Assert.NotEqual(t1, t3);

        var t4 = new Tensor<int>(momory, new int[] { 1, 1, 2, 4 });
        Assert.Equal(t1, t4);
    }

    [Fact]
    public void TestEquals2()
    {
        var array = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var momory = new Memory<int>(array);

        var t1 = new Tensor<int>(momory, new int[] { 1, 1, 2, 4 });
        var t2 = new Tensor<int>(momory, new int[] { 2, 4 });
        Assert.False(t1.Equals((object)t2));

        var t3 = new Tensor<int>(momory, new int[] { 1, 1, 4, 2 });
        Assert.False(t1.Equals((object)t3));

        var t4 = new Tensor<int>(momory, new int[] { 1, 1, 2, 4 });
        Assert.True(t1.Equals((object)t4));

        Assert.Throws<ArgumentException>(() => t1.Equals((object)array));
    }
}
