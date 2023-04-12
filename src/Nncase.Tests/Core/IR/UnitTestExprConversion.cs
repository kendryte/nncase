// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Linq.Expressions;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using Nncase.IR;
using Xunit;

namespace Nncase.Tests.CoreTest;

public class UnitTestExprConversion
{
    [Fact]
    public void TestByte()
    {
        byte b = 1;
        Const c = b;
        Expr e = b;
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestUshort()
    {
        ushort b = 1;
        Const c = b;
        Expr e = b;
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestUint()
    {
        uint b = 1;
        Const c = b;
        Expr e = b;
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestUlong()
    {
        ulong b = 1;
        Const c = b;
        Expr e = b;
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestSbyte()
    {
        sbyte b = 1;
        Const c = b;
        Expr e = b;
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestShort()
    {
        short b = 1;
        Const c = b;
        Expr e = b;
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestInt()
    {
        int b = 1;
        Const c = b;
        Expr e = b;
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestLong()
    {
        long b = 1;
        Const c = b;
        Expr e = b;
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestHalf()
    {
        var b = (Half)1F;
        Const c = b;
        Expr e = b;
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestFloat()
    {
        float b = 1F;
        Const c = b;
        Expr e = b;
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestDouble()
    {
        double b = 1;
        Const c = b;
        Expr e = b;
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestBFloat16()
    {
        var b = (BFloat16)1F;
        Const c = b;
        Expr e = b;
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestBool()
    {
        var b = true;
        Const c = b;
        Expr e = b;
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestShape()
    {
        var a = new int[] { 1, 3, 16, 16 };
        var s = new Shape(a);
        var c = Const.FromShape(s);
        Expr e = s;
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestIntArray()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        Expr e = a;
        var c = Const.FromTensor(Tensor.From<int>(a));
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestLongArray()
    {
        var a = new long[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        Expr e = a;
        var c = Const.FromTensor(Tensor.From<long>(a));
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestFloatArray()
    {
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        Expr e = a;
        var c = Const.FromTensor(Tensor.From<float>(a));
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestArray()
    {
        var a = Array.CreateInstance(typeof(float), 8);
        Expr e = a;
        var c = Const.FromTensor(Tensor.FromArray(a));
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestIntMemory()
    {
        var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var m = new Memory<int>(a);
        Expr e = m;
        var c = Const.FromTensor(Tensor.From<int>(m));
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestLongMemory()
    {
        var a = new long[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var m = new Memory<long>(a);
        Expr e = m;
        var c = Const.FromTensor(Tensor.From<long>(m));
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestFloatMemory()
    {
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var m = new Memory<float>(a);
        Expr e = m;
        var c = Const.FromTensor(Tensor.From<float>(m));
        Assert.Equal(c, e);
    }

    [Fact]
    public void TestQuantParam()
    {
        var qp = new QuantParam(0, 1);
        var c = Const.FromTensor(Tensor.FromScalar(qp));
        Expr e = qp;
        Assert.Equal(c, e);
    }
}
