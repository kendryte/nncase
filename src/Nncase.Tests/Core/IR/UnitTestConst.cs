// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using Nncase;
using Nncase.IR;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestConst
{
    [Fact]
    public void TestByte()
    {
        byte expected = 1;
        Const c = expected;
        var tc = (TensorConst)c;
        Assert.True(tc.CheckedShape.IsScalar);
        Assert.Equal(DataType.FromType<byte>(), tc.Value.ElementType);
        var list = (IList)tc.Value;
        Assert.Equal(expected, list[0]);
    }

    [Fact]
    public void TestUshort()
    {
        ushort expected = 1;
        Const c = expected;
        var tc = (TensorConst)c;
        Assert.True(tc.CheckedShape.IsScalar);
        Assert.Equal(DataType.FromType<ushort>(), tc.Value.ElementType);
        var list = (IList)tc.Value;
        Assert.Equal(expected, list[0]);
    }

    [Fact]
    public void TestUint()
    {
        uint expected = 1;
        Const c = expected;
        var tc = (TensorConst)c;
        Assert.True(tc.CheckedShape.IsScalar);
        Assert.Equal(DataType.FromType<uint>(), tc.Value.ElementType);
        var list = (IList)tc.Value;
        Assert.Equal(expected, list[0]);
    }

    [Fact]
    public void TestUlong()
    {
        ulong expected = 1;
        Const c = expected;
        var tc = (TensorConst)c;
        Assert.True(tc.CheckedShape.IsScalar);
        Assert.Equal(DataType.FromType<ulong>(), tc.Value.ElementType);
        var list = (IList)tc.Value;
        Assert.Equal(expected, list[0]);
    }

    [Fact]
    public void TestSbyte()
    {
        sbyte expected = 1;
        Const c = expected;
        var tc = (TensorConst)c;
        Assert.True(tc.CheckedShape.IsScalar);
        Assert.Equal(DataType.FromType<sbyte>(), tc.Value.ElementType);
        var list = (IList)tc.Value;
        Assert.Equal(expected, list[0]);
    }

    [Fact]
    public void TestShort()
    {
        short expected = 1;
        Const c = expected;
        var tc = (TensorConst)c;
        Assert.True(tc.CheckedShape.IsScalar);
        Assert.Equal(DataType.FromType<short>(), tc.Value.ElementType);
        var list = (IList)tc.Value;
        Assert.Equal(expected, list[0]);
    }

    [Fact]
    public void TestInt()
    {
        int expected = 1;
        Const c = expected;
        var tc = (TensorConst)c;
        Assert.True(tc.CheckedShape.IsScalar);
        Assert.Equal(DataType.FromType<int>(), tc.Value.ElementType);
        var list = (IList)tc.Value;
        Assert.Equal(expected, list[0]);
    }

    [Fact]
    public void TestLong()
    {
        long expected = 1;
        Const c = expected;
        var tc = (TensorConst)c;
        Assert.True(tc.CheckedShape.IsScalar);
        Assert.Equal(DataType.FromType<long>(), tc.Value.ElementType);
        var list = (IList)tc.Value;
        Assert.Equal(expected, list[0]);
    }

    [Fact]
    public void TestHalf()
    {
        var expected = (Half)1F;
        Const c = expected;
        var tc = (TensorConst)c;
        Assert.True(tc.CheckedShape.IsScalar);
        Assert.Equal(DataType.FromType<Half>(), tc.Value.ElementType);
        var list = (IList)tc.Value;
        Assert.Equal(expected, list[0]);
    }

    [Fact]
    public void TestFloat()
    {
        var expected = 1F;
        Const c = expected;
        var tc = (TensorConst)c;
        Assert.True(tc.CheckedShape.IsScalar);
        Assert.Equal(DataType.FromType<float>(), tc.Value.ElementType);
        var list = (IList)tc.Value;
        Assert.Equal(expected, list[0]);
    }

    [Fact]
    public void TestDouble()
    {
        var expected = 1D;
        Const c = expected;
        var tc = (TensorConst)c;
        Assert.True(tc.CheckedShape.IsScalar);
        Assert.Equal(DataType.FromType<double>(), tc.Value.ElementType);
        var list = (IList)tc.Value;
        Assert.Equal(expected, list[0]);
    }

    [Fact]
    public void TestBfloat16()
    {
        var expected = (BFloat16)1F;
        Const c = expected;
        var tc = (TensorConst)c;
        Assert.True(tc.CheckedShape.IsScalar);
        Assert.Equal(DataType.FromType<BFloat16>(), tc.Value.ElementType);
        var list = (IList)tc.Value;
        Assert.Equal(expected, list[0]);
    }

    [Fact]
    public void TestBool()
    {
        var expected = false;
        Const c = expected;
        var tc = (TensorConst)c;
        Assert.True(tc.CheckedShape.IsScalar);
        Assert.Equal(DataType.FromType<bool>(), tc.Value.ElementType);
        var list = (IList)tc.Value;
        Assert.Equal(expected, list[0]);
    }

    [Fact]
    public void TestUtf8Char()
    {
        byte b = 1;
        Utf8Char expected = b;
        Const c = expected;
        var tc = (TensorConst)c;
        Assert.True(tc.CheckedShape.IsScalar);
        Assert.Equal(DataType.FromType<Utf8Char>(), tc.Value.ElementType);
        var list = (IList)tc.Value;
        Assert.Equal(expected, list[0]);
    }

    [Fact]
    public void TestFromTensorValue()
    {
        byte b = 1;
        var tv = new TensorValue(b);
        var c = Const.FromValue(tv);
        var tc = (TensorConst)c;
        Assert.True(tc.CheckedShape.IsScalar);
        Assert.Equal(DataType.FromType<byte>(), tc.Value.ElementType);
        var list = (IList)tc.Value;
        Assert.Equal(b, list[0]);
    }

    [Fact]
    public void TestFromTupleValue()
    {
        var dims = new int[] { 1, 3, 16, 16 };
        var t1 = Tensor.Ones<float>(dims);
        var t2 = Tensor.Zeros<float>(dims);
        var tensors = new Tensor[] { t1, t2 };
        var tpv = Value.FromTensors(tensors);

        var c = Const.FromValue(tpv);
        var tpc = (TupleConst)c;
        Assert.Equal(tensors.Length, tpc.Count);
    }
}
