// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Immutable;
using Nncase;
using Nncase.IR;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestTensor
{
    [Fact]
    public void TestICollection()
    {
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t = (ICollection)Tensor.From(a, new int[] { 1, 1, 2, 4 });
        Assert.Equal(a.Length, t.Count);
        Assert.False(t.IsSynchronized);
        Assert.Equal((object)t, t.SyncRoot);
    }

    [Fact]
    public void TestIList()
    {
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t = Tensor.From(a, new int[] { 1, 1, 2, 4 });
        var list = (IList)t;
        Assert.True(list.IsFixedSize);
        Assert.False(list.IsReadOnly);

        Assert.Equal(1f, list[0]);
        list[0] = 100f;
        Assert.Equal(100f, list[0]);

        list.Clear();
        var expected = Tensor.Zeros<float>(new int[] { 1, 1, 2, 4 });
        Assert.Equal(expected, t);
    }

    [Fact]
    public void TestIndices()
    {
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t = Tensor.From(a, new int[] { 1, 1, 2, 4 });

        Assert.Equal(7, t[new int[] { 0, 0, 1, 2 }]);
        t[new int[] { 0, 0, 1, 2 }] = 700;
        Assert.Equal(700, t[new int[] { 0, 0, 1, 2 }]);
    }

    // Tensor<T> FromBytes<T>(Memory<byte> memory, ReadOnlySpan<int> dimensions)
    [Fact]
    public void TestFromBytesOverload1()
    {
        var a = new byte[] { 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x40 };
        var expected = new float[] { 1, 2, 3, 4 };
        var t = Tensor.FromBytes<float>(new Memory<byte>(a), new int[] { 1, 1, 2, 2 });
        Assert.Equal(DataTypes.Float32, t.ElementType);
        Assert.Equal(expected, t.ToArray<float>());
    }

    // Tensor FromBytes(DataType type, Memory<byte> memory, ReadOnlySpan<int> dimensions)
    [Fact]
    public void TestFromBytesOverload2()
    {
        var a = new byte[] { 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x40 };
        var expected = new float[] { 1, 2, 3, 4 };
        var t = Tensor.FromBytes(DataTypes.Float32, new Memory<byte>(a), new int[] { 1, 1, 2, 2 });
        Assert.Equal(DataTypes.Float32, t.ElementType);
        Assert.Equal(expected, t.ToArray<float>());
    }

    // Tensor FromBytes(TensorType type, Memory<byte> buffer)
    [Fact]
    public void TestFromBytesOverload3()
    {
        var a = new byte[] { 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x40 };
        var expected = new float[] { 1, 2, 3, 4 };
        var tensorType = new TensorType(DataTypes.Float32, new int[] { 1, 1, 2, 2 });
        var t = Tensor.FromBytes(tensorType, new Memory<byte>(a));
        Assert.Equal(DataTypes.Float32, t.ElementType);
        Assert.Equal(expected, t.ToArray<float>());
    }

    [Fact]
    public unsafe void TestFromPointer()
    {
        var value1 = 2022;
        var value2 = 2023;
        var addr1 = (ulong)&value1;
        var addr2 = (ulong)&value2;
        var p1 = new Pointer<int>(addr1);
        var p2 = new Pointer<int>(addr2);

        var t = Tensor.FromPointer<int>(addr1);
        Assert.Equal(p1, t.ToScalar<Pointer<int>>());
        Assert.Equal(addr1, t.ToScalar<Pointer<int>>().Value);
        Assert.NotEqual(p2, t.ToScalar<Pointer<int>>());
    }

    [Fact]
    public unsafe void TestFromPointerET()
    {
        var value1 = 2022;
        var value2 = 2023;
        var addr1 = (ulong)&value1;
        var addr2 = (ulong)&value2;
        var p1 = new Pointer<int>(addr1);
        var p2 = new Pointer<int>(addr2);

        var t = Tensor.FromPointer(addr1, new PointerType(DataTypes.Int32, Shape.Scalar));
        Assert.Equal(p1, t.ToScalar<Pointer<int>>());
        Assert.Equal(addr1, t.ToScalar<Pointer<int>>().Value);
        Assert.NotEqual(p2, t.ToScalar<Pointer<int>>());
    }

    // Tensor FromConst(Const @const)
    [Fact]
    public unsafe void TestFromConstOverload1()
    {
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t = Tensor.From(a, new int[] { 1, 1, 2, 4 });
        var tensorConst1 = new TensorConst(t);
        var tensorConst2 = tensorConst1;

        // TensorConst
        Assert.Equal(t, Tensor.FromConst(tensorConst1));

        // TupleConst
        var tupleConst = new TupleConst(Value.FromTensors(tensorConst1.Value, tensorConst2.Value));
        Assert.Throws<InvalidOperationException>(() => Tensor.FromConst(tupleConst));
    }

    // Tensor<T> FromConst<T>(Const @const, CastMode castMode = CastMode.KDefault)
    [Fact]
    public unsafe void TestFromConstOverload2()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t1 = Tensor.From(a, new int[] { 1, 1, 2, 4 });
        var tensorConst1 = new TensorConst(t1);

        var expected = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t2 = Tensor.FromConst<float>(tensorConst1);
        Assert.Equal(expected, t2);
    }

    [Fact]
    public void TestListException()
    {
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t = (IList)Tensor.From(a, new int[] { 1, 1, 2, 4 });

        Assert.Throws<InvalidOperationException>(() => t.Add(100));
        Assert.Throws<InvalidOperationException>(() => t.Insert(0, 100));
        Assert.Throws<InvalidOperationException>(() => t.Remove(8));
        Assert.Throws<InvalidOperationException>(() => t.RemoveAt(7));
        Assert.Throws<NotImplementedException>(() => t.Contains(8));
        Assert.Throws<NotImplementedException>(() => t.IndexOf(8));
    }
}
