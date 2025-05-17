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
        var t = new Tensor<int>(memory, [1, 1, 2, 4]);
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
        var t = new Tensor<int>(memory, [8]);
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
        var t = new Tensor<int>(memory, [8]);
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
        Assert.Equal(scalar, a[Array.Empty<long>()]);
    }

    [Fact]
    public void TestClone()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);
        var t1 = new Tensor<int>(memory, [8]);
        var t2 = t1.Clone();
        Assert.Equal(t1, t2);
        Assert.NotSame(t1, t2);
    }

    [Fact]
    public void TestCloneEmpty()
    {
        var t1 = new Tensor<float>([1, 2, 3, 4]);
        var t2 = t1.CloneEmpty<int>([1, 3, 16, 16]);
        var a = (ICollection<int>)t2;
        Assert.Equal(1 * 3 * 16 * 16, a.Count);
    }

    [Fact]
    public void TestGetValue()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);
        var t = new Tensor<int>(memory, [2, 4]);
        Assert.Equal(2, t.GetValue(1));
        Assert.Equal(7, t.GetValue(6));
        Assert.Throws<IndexOutOfRangeException>(() => t.GetValue(a.Length));
    }

    [Fact]
    public void TestReshape()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);

        var t1 = new Tensor<int>(memory, [1, 1, 2, 4]);
        Assert.Throws<ArgumentException>(() => t1.Reshape([1, 1, 4, 3]));

        var t2 = t1.Reshape([1, 1, 4, 2]);
        Assert.True(t2.Buffer.Span.SequenceEqual(t1.Buffer.Span));
    }

    [Theory]
    [MemberData(nameof(TestSetValueData))]
    public void TestSetValue(int index, int value)
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);
        var t = new Tensor<int>(memory, [2, 4]);
        t.SetValue(index, value);
        Assert.Equal(value, t.GetValue(index));
        Assert.Throws<IndexOutOfRangeException>(() => t.SetValue(a.Length, 9));
    }

    [Fact]
    public void TestFill()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);
        var t = new Tensor<int>(memory, [2, 4]);
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
        var t = new Tensor<int>(memory, [2, 4]);
        Assert.Contains(1, t);
        Assert.Contains(8, t);
    }

    [Fact]
    public void TestTryGetIndicesOf()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);
        var t = new Tensor<int>(memory, [1, 1, 2, 4]);

        var indices1 = new long[] { 1, 1, 1, 1 };
        Assert.True(t.TryGetIndicesOf(7, indices1));
        Assert.Equal(0, indices1[0]);
        Assert.Equal(0, indices1[1]);
        Assert.Equal(1, indices1[2]);
        Assert.Equal(2, indices1[3]);

        var indices2 = new long[] { 1, 1 };
        Assert.Throws<ArgumentException>(() => t.TryGetIndicesOf(7, indices2));

        var indices3 = new long[] { 1, 1, 1, 1 };
        Assert.False(t.TryGetIndicesOf(100, indices3));
    }

    // bool Equals(object? obj)
    [Fact]
    public void TestEqualsOverload1()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);

        var t1 = new Tensor<int>(memory, [1, 1, 2, 4]);
        var t2 = new Tensor<int>(memory, [2, 4]);
        Assert.False(t1.Equals((object)t2));

        var t3 = new Tensor<int>(memory, [1, 1, 4, 2]);
        Assert.False(t1.Equals((object)t3));

        var t4 = new Tensor<int>(memory, [1, 1, 2, 4]);
        Assert.True(t1.Equals((object)t4));

        Assert.False(t1.Equals(a));
    }

    // bool Equals(Tensor<T>? other)
    [Fact]
    public void TestEqualsOverload2()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var memory = new Memory<int>(a);

        var t1 = new Tensor<int>(memory, [1, 1, 2, 4]);
        Tensor<int>? n = null;
        Assert.NotStrictEqual(t1, n);
        Assert.False(t1.Equals(n));

        var t2 = new Tensor<int>(memory, [2, 4]);
        Assert.NotStrictEqual(t1, t2);

        var t3 = new Tensor<int>(memory, [1, 1, 4, 2]);
        Assert.NotStrictEqual(t1, t3);

        var t4 = new Tensor<int>(memory, [1, 1, 2, 4]);
        Assert.StrictEqual(t1, t4);
    }

    [Fact]
    public void TestEmpty()
    {
        var t1 = new Tensor<float>([0]);
        Assert.Equal(0, t1.Length);
    }

    [Fact]
    public void TestTensorGetString()
    {
        var a = Tensor.From(new float[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, [4, 4]);
        var expect = @"{
  [0,0]: {0f,1f,2f,3f},
  [1,0]: {4f,5f,6f,7f},
  [2,0]: {8f,9f,10f,11f},
  [3,0]: {12f,13f,14f,15f}
}";
        Assert.Equal(expect, a.GetArrayString(true));

        var b = Tensor.From(new float[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, [1, 4, 4]);
        var expectb = @"{
  {
    [0,0,0]: {0f,1f,2f,3f},
    [0,1,0]: {4f,5f,6f,7f},
    [0,2,0]: {8f,9f,10f,11f},
    [0,3,0]: {12f,13f,14f,15f}
  }
}";
        Assert.Equal(expectb, b.GetArrayString(true));
    }

    [Fact]
    public void TestVectorTensor()
    {
        var a = Tensor<Vector4<float>>.From(new Vector4<float>[] { Vector4<float>.Create([1, 2, 3, 4]), Vector4<float>.Create([1, 2, 3, 4]), Vector4<float>.Create([1, 2, 3, 4]), Vector4<float>.Create([1, 2, 3, 4]) });
        Assert.Equal(a.ElementType, new VectorType(DataTypes.Float32, [4]));

        var b = Enumerable.Range(0, 12).Select(i => i).ToArray();
        var d = Tensor<Vector4<float>>.From(System.Runtime.InteropServices.MemoryMarshal.Cast<int, Vector4<int>>(b).ToArray());
        Assert.Equal(d.ElementType, new VectorType(DataTypes.Int32, [4]));
        Assert.Equal("{<0,1,2,3>,<4,5,6,7>,<8,9,10,11>}", d.GetArrayString(false));
    }

    [Fact]
    public void TestTensorView()
    {
        {
            var a = Tensor.From(Enumerable.Range(0, 2 * 2 * 3).ToArray(), [2, 2, 3]);
            /*
            [[[ 0,  1,  2],
              [ 3,  4,  5]],
             [[ 6,  7,  8],
              [ 9, 10, 11]]]
            */
            var b = a.View([1, 0, 0], [1, 1, 3]); /* [6,7,8] */
            Assert.True(b.ToArray<int>().SequenceEqual([6, 7, 8]));
            Assert.Equal("{{{6,7,8}}}", b.GetArrayString(false));
            Assert.Equal("{6,7,8}", b.Squeeze([0, 1]).GetArrayString(false));

            var c = new List<int>();
            c.AddRange(b);
            Assert.True(c.SequenceEqual([6, 7, 8]));

            var d = new List<int>();
            foreach (var item in b)
            {
                d.Add(item);
            }

            Assert.True(d.SequenceEqual([6, 7, 8]));
        }

        {
            var a = Tensor.From(Enumerable.Range(0, 4 * 3).ToArray(), [4, 3]);
            var b = a.View([0, 0], [4, 1]);
            Assert.True(b.ToArray<int>().SequenceEqual([0, 3, 6, 9]));
            Assert.Equal("{{0},{3},{6},{9}}", b.GetArrayString(false));
            Assert.Equal("{0,3,6,9}", b.Squeeze([1]).GetArrayString(false));
        }
    }

    [Fact]
    public void TestTensorViewCopy()
    {
        {
            var a = Tensor.From(Enumerable.Range(0, 2 * 2 * 3).ToArray(), [2, 2, 3]);
            var b = a.View([0, 1, 0], [1, 1, 3]); /* [[[3,4,5]]] */
            Assert.True(b.ToArray<int>().SequenceEqual([3, 4, 5]));

            var c = Tensor.Zeros<int>([3]);
            Assert.Throws<ArgumentException>(() =>
            {
                b.CopyTo(c);
            });

            b.Squeeze([0, 1]).CopyTo(c);
            Assert.True(c.ToArray<int>().SequenceEqual([3, 4, 5]));
        }

        {
            var a = Tensor.From(Enumerable.Range(0, 2 * 2 * 3).Select(i => Vector4<int>.Create([i, i, i, i])).ToArray(), [2, 2, 3]);

            var b = a.View([1, 1, 0], [1, 1, 3]); /* [[[<9,9,9,9>,<10,...>,<11,...>]]] */
            Assert.True(b.ToArray<Vector4<int>>().SequenceEqual([Vector4<int>.Create([9, 9, 9, 9]), Vector4<int>.Create([10, 10, 10, 10]), Vector4<int>.Create([11, 11, 11, 11])]));
        }
    }

    [Fact]
    public void TestTensorTranspose()
    {
        {
            var a = Tensor.From(new[] { 1, 2, 3, 4, 5, 6 }, [2, 3]);
            var b = a.Transpose([1, 0]);
            Assert.Equal("{{1,4},{2,5},{3,6}}", b.GetArrayString(false));
        }

        {
            var a = Tensor.From(Enumerable.Range(0, 24).ToArray(), [2, 3, 4]);
            var b = a.Transpose([2, 0, 1]);
            Assert.Equal(4, b.Dimensions[0]);
            Assert.Equal(2, b.Dimensions[1]);
            Assert.Equal(3, b.Dimensions[2]);
            var expected = new[] {
                0,  4,  8, 12, 16, 20,  1,  5,  9, 13, 17, 21,  2,  6, 10, 14, 18, 22,  3,  7, 11, 15, 19, 23,
            };
            Assert.True(b.ToArray<int>().SequenceEqual(expected));
        }

        {
            var a = Tensor.From(Enumerable.Range(0, 16).ToArray(), [2, 2, 2, 2]);
            var b = a.Transpose([3, 2, 1, 0]);
            Assert.Equal(2, b.Dimensions[0]);
            Assert.Equal(2, b.Dimensions[1]);
            Assert.Equal(2, b.Dimensions[2]);
            Assert.Equal(2, b.Dimensions[3]);
        }

        {
            var a = Tensor.From(new[] { 1, 2, 3, 4 }, [2, 2]);
            Assert.Throws<ArgumentException>(() => a.Transpose([0, 1, 2]));
        }

        {
            var a = Tensor.From(new[] { 1, 2, 3, 4 }, [2, 2]);
            Assert.Throws<ArgumentException>(() => a.Transpose([0, 2]));
        }

        {
            var vec = Vector4<float>.Create([1, 2, 3, 4]);
            var a = Tensor.From(new[] { vec, vec }, [2, 1]);
            var b = a.Transpose([1, 0]);
            Assert.Equal(1, b.Dimensions[0]);
            Assert.Equal(2, b.Dimensions[1]);
        }
    }
}
