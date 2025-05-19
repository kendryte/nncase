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

public class UnitTestExprOperators
{
    [Fact]
    public void TestGetItemOfTensorWithScalarIndex()
    {
        var a = (Expr)new[] { 1, 2, 3 };
        Assert.Equal((Expr)1, a[0]);
        Assert.Equal((Expr)2, a[1]);
        Assert.Equal((Expr)3, a[2]);
    }

    [Fact]
    public void TestGetItemOfTensorWithScalarExprIndex()
    {
        var a = (Expr)new[] { 1, 2, 3 };
        Assert.NotEqual(1, a[(Dimension)0]);
        Assert.NotEqual(2, a[(Dimension)1]);
        Assert.NotEqual(3, a[(Dimension)2]);
    }

    [Fact]
    public void TestGetItemOfTupleWithScalarIndex()
    {
        var a = new IR.Tuple([(Expr)1, (Expr)2, (Expr)3]);
        Assert.Equal((Expr)1, a[0]);
        Assert.Equal((Expr)2, a[1]);
        Assert.Equal((Expr)3, a[2]);
    }

    [Fact]
    public void TestGetItemOfStackWithScalarIndex()
    {
        var a = new IR.Tuple([(Expr)1, (Expr)2, (Expr)3]);
        var b = IR.F.Tensors.Stack(a, 0);
        Assert.True(b[0] is Call);
        Assert.True(b[1] is Call);
        Assert.True(b[2] is Call);
    }

    [Fact]
    public void TestGetItemOfConcatWithScalarIndex1()
    {
        var a = Enumerable.Range(1, 3).Select(x => (Expr)new[] { x }).ToArray();
        var b = IR.F.Tensors.Concat(new IR.Tuple(a), 0);
        Assert.True(b[0] is Call);
        Assert.True(b[1] is Call);
        Assert.True(b[2] is Call);
    }

    [Fact]
    public void TestGetItemOfConcatWithScalarIndex2()
    {
        var a = Enumerable.Range(1, 3).Select(x => (Expr)new[] { x, x }).ToArray();
        var b = IR.F.Tensors.Concat(new IR.Tuple(a), 0);
        Assert.True(b[0] is Call);
        Assert.True(b[1] is Call);
        Assert.True(b[2] is Call);
        Assert.True(b[3] is Call);
    }
}
