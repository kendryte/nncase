// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestTensorConversions
{
    [Fact]
    public void TestTensorByte()
    {
        byte a = 1;
        var t = (Tensor)a;
        Assert.Equal(1, t.Length);
        Assert.Equal(a, t.ToScalar<byte>());
    }

    [Fact]
    public void TestTensorUshort()
    {
        ushort a = 1;
        var t = (Tensor)a;
        Assert.Equal(1, t.Length);
        Assert.Equal(a, t.ToScalar<ushort>());
    }

    [Fact]
    public void TestTensorUint()
    {
        uint a = 1;
        var t = (Tensor)a;
        Assert.Equal(1, t.Length);
        Assert.Equal(a, t.ToScalar<uint>());
    }

    [Fact]
    public void TestTensorUlong()
    {
        ulong a = 1;
        var t = (Tensor)a;
        Assert.Equal(1, t.Length);
        Assert.Equal(a, t.ToScalar<ulong>());
    }

    [Fact]
    public void TestTensorSbyte()
    {
        sbyte a = 1;
        var t = (Tensor)a;
        Assert.Equal(1, t.Length);
        Assert.Equal(a, t.ToScalar<sbyte>());
    }

    [Fact]
    public void TestTensorShort()
    {
        short a = 1;
        var t = (Tensor)a;
        Assert.Equal(1, t.Length);
        Assert.Equal(a, t.ToScalar<short>());
    }

    [Fact]
    public void TestTensorInt()
    {
        int a = 1;
        var t = (Tensor)a;
        Assert.Equal(1, t.Length);
        Assert.Equal(a, t.ToScalar<int>());
    }

    [Fact]
    public void TestTensorLong()
    {
        long a = 1;
        var t = (Tensor)a;
        Assert.Equal(1, t.Length);
        Assert.Equal(a, t.ToScalar<long>());
    }

    [Fact]
    public void TestTensorHalf()
    {
        var a = (Half)1.0F;
        var t = (Tensor)a;
        Assert.Equal(1, t.Length);
        Assert.Equal(a, t.ToScalar<Half>());
    }

    [Fact]
    public void TestTensorFloat()
    {
        var a = 1.0F;
        var t = (Tensor)a;
        Assert.Equal(1, t.Length);
        Assert.Equal(a, t.ToScalar<float>());
    }

    [Fact]
    public void TestTensorDouble()
    {
        var a = 1.0D;
        var t = (Tensor)a;
        Assert.Equal(1, t.Length);
        Assert.Equal(a, t.ToScalar<double>());
    }

    [Fact]
    public void TestTensorBFloat16()
    {
        var a = (BFloat16)1.0F;
        var t = (Tensor)a;
        Assert.Equal(1, t.Length);
        Assert.Equal(a, t.ToScalar<BFloat16>());
    }

    [Fact]
    public void TestTensorBool()
    {
        var a = true;
        var t = (Tensor)a;
        Assert.Equal(1, t.Length);
        Assert.Equal(a, t.ToScalar<bool>());
    }

    [Fact]
    public void TestTensorMemoryByte()
    {
        var a = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t = (Tensor)new Memory<byte>(a);
        Assert.Equal(a.Length, t.Length);
        Assert.Equal(a, t.ToArray<byte>());
    }

    [Fact]
    public void TestTensorMemoryUshort()
    {
        var a = new ushort[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t = (Tensor)new Memory<ushort>(a);
        Assert.Equal(a.Length, t.Length);
        Assert.Equal(a, t.ToArray<ushort>());
    }

    [Fact]
    public void TestTensorMemoryUint()
    {
        var a = new uint[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t = (Tensor)new Memory<uint>(a);
        Assert.Equal(a.Length, t.Length);
        Assert.Equal(a, t.ToArray<uint>());
    }

    [Fact]
    public void TestTensorMemoryUlong()
    {
        var a = new ulong[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t = (Tensor)new Memory<ulong>(a);
        Assert.Equal(a.Length, t.Length);
        Assert.Equal(a, t.ToArray<ulong>());
    }

    [Fact]
    public void TestTensorMemorySbyte()
    {
        var a = new sbyte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t = (Tensor)new Memory<sbyte>(a);
        Assert.Equal(a.Length, t.Length);
        Assert.Equal(a, t.ToArray<sbyte>());
    }

    [Fact]
    public void TestTensorMemoryShort()
    {
        var a = new short[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t = (Tensor)new Memory<short>(a);
        Assert.Equal(a.Length, t.Length);
        Assert.Equal(a, t.ToArray<short>());
    }

    [Fact]
    public void TestTensorMemoryInt()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t = (Tensor)new Memory<int>(a);
        Assert.Equal(a.Length, t.Length);
        Assert.Equal(a, t.ToArray<int>());
    }

    [Fact]
    public void TestTensorMemoryLong()
    {
        var a = new long[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t = (Tensor)new Memory<long>(a);
        Assert.Equal(a.Length, t.Length);
        Assert.Equal(a, t.ToArray<long>());
    }

    [Fact]
    public void TestTensorMemoryHalf()
    {
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var b = a.Select(x => (Half)x).ToArray<Half>();
        var t = (Tensor)new Memory<Half>(b);
        Assert.Equal(b.Length, t.Length);
        Assert.Equal(b, t.ToArray<Half>());
    }

    [Fact]
    public void TestTensorMemoryFloat()
    {
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t = (Tensor)new Memory<float>(a);
        Assert.Equal(a.Length, t.Length);
        Assert.Equal(a, t.ToArray<float>());
    }

    [Fact]
    public void TestTensorMemoryDouble()
    {
        var a = new double[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t = (Tensor)new Memory<double>(a);
        Assert.Equal(a.Length, t.Length);
        Assert.Equal(a, t.ToArray<double>());
    }

    [Fact]
    public void TestTensorMemoryBFloat16()
    {
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var b = a.Select(x => (BFloat16)x).ToArray<BFloat16>();
        var t = (Tensor)new Memory<BFloat16>(b);
        Assert.Equal(b.Length, t.Length);
        Assert.Equal(b, t.ToArray<BFloat16>());
    }

    [Fact]
    public void TestTensorMemoryBool()
    {
        var a = new bool[] { true, false, false, true, true, false, false, true };
        var t = (Tensor)new Memory<bool>(a);
        Assert.Equal(a.Length, t.Length);
        Assert.Equal(a, t.ToArray<bool>());
    }

    [Fact]
    public void TestTensorArray()
    {
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var t = (Tensor)a;
        Assert.Equal(a.Length, t.Length);
        Assert.Equal(a, t.ToArray<float>());
    }

    [Fact]
    public void TestTensorValue()
    {
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var v = (TensorValue)(Tensor)a;
        Assert.Single(v.AsTensors());
        Assert.Equal(a, v.AsTensor().ToArray<float>());
    }
}
