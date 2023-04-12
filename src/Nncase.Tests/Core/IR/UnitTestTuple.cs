// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using Nncase;
using Nncase.IR;
using Nncase.Tests.TestFixture;
using Xunit;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestTuple
{
    [Fact]
    public void TestVoid()
    {
        var v = Tuple.Void;
        Assert.Empty(v.Value);
    }

    [Fact]
    public void TestConstructor1()
    {
        byte b1 = 1;
        byte b2 = 2;

        TensorConst tc1 = b1;
        TensorConst tc2 = b2;
        var a = new Expr[] { tc1, tc2 };
        var tp = new Tuple(a);
        Assert.Equal(a.Length, tp.Count);
    }

    [Fact]
    public void TestConstructor2()
    {
        byte b1 = 1;
        byte b2 = 2;

        TensorConst tc1 = b1;
        TensorConst tc2 = b2;
        var list = new List<Expr>();
        list.Add(tc1);
        list.Add(tc2);
        var tp = new Tuple(list.ToArray());
        Assert.Equal(list.Count, tp.Count);
    }

    [Fact]
    public void TestFields()
    {
        byte b1 = 1;
        byte b2 = 2;

        TensorConst tc1 = b1;
        TensorConst tc2 = b2;
        var a = new Expr[] { tc1, tc2 };
        var tp = new Tuple(a);

        var itp = (ITuple)tp;
        var fields = tp.Fields;
        Assert.Equal(a.Length, itp.Count);
        Assert.Equal(tc1, fields[0]);
        Assert.Equal(tc2, fields[1]);
    }

    [Fact]
    public void TestIndex()
    {
        byte b1 = 1;
        byte b2 = 2;
        TensorConst tc1 = b1;
        TensorConst tc2 = b2;
        var a = new Expr[] { tc1, tc2 };
        var tp = new Tuple(a);
        Assert.Equal(tc1, tp[0]);
        Assert.Equal(tc2, tp[1]);
    }

    [Fact]
    public void TestImplicitOperatorOverload1()
    {
        byte b = 1;
        TensorConst tc = b;
        var vt = ValueTuple.Create(tc);
        Tuple tp = vt;
        Assert.Equal(tc, tp[0]);
    }

    [Fact]
    public void TestImplicitOperatorOverload2()
    {
        byte b1 = 1;
        byte b2 = 2;
        TensorConst tc1 = b1;
        TensorConst tc2 = b2;
        Tuple tp = (tc1, tc2);
        Assert.Equal(tc1, tp[0]);
        Assert.Equal(tc2, tp[1]);
    }

    [Fact]
    public void TestImplicitOperatorOverload3()
    {
        byte b1 = 1;
        byte b2 = 2;
        byte b3 = 3;
        TensorConst tc1 = b1;
        TensorConst tc2 = b2;
        TensorConst tc3 = b3;
        Tuple tp = (tc1, tc2, tc3);
        Assert.Equal(tc1, tp[0]);
        Assert.Equal(tc2, tp[1]);
        Assert.Equal(tc3, tp[2]);
    }

    [Fact]
    public void TestImplicitOperatorOverload4()
    {
        byte b1 = 1;
        byte b2 = 2;
        byte b3 = 3;
        byte b4 = 4;
        TensorConst tc1 = b1;
        TensorConst tc2 = b2;
        TensorConst tc3 = b3;
        TensorConst tc4 = b4;
        Tuple tp = (tc1, tc2, tc3, tc4);
        Assert.Equal(tc1, tp[0]);
        Assert.Equal(tc2, tp[1]);
        Assert.Equal(tc3, tp[2]);
        Assert.Equal(tc4, tp[3]);
    }
}
