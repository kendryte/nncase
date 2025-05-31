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

public sealed class UnitTestDimension
{
    [Fact]
    public void TestValue()
    {
        long v1 = 1;
        var d1 = new DimConst(v1);
        Assert.Equal(v1, d1.Value);
        Assert.Equal(v1, d1.FixedValue);

        long v2 = -1;
        var d2 = new DimConst(v2);
        Assert.Equal(v2, d2.Value);
        Assert.Equal(v2, d2.FixedValue);
    }

    [Fact]
    public void TestKind()
    {
        long v1 = 1;
        var d1 = new DimConst(v1);
        Assert.Equal(DimensionKind.Fixed, d1.Kind);
        Assert.False(d1.IsUnknown);
        Assert.True(d1.IsFixed);

        var d2 = Dimension.Unknown;
        Assert.Equal(DimensionKind.Unknown, d2.Kind);
        Assert.True(d2.IsUnknown);
        Assert.False(d2.IsFixed);
    }

    [Fact]
    public void TestOperatorEqual()
    {
        Dimension d1 = 1;
        Dimension d2 = 1;
        Dimension d3 = 3;
        Assert.True(d1 == d2);
        Assert.False(d1 == d3);
    }

    [Fact]
    public void TestOperatorNotEqual()
    {
        Dimension d1 = 1;
        Dimension d2 = 1;
        Dimension d3 = 3;
        Assert.False(d1 != d2);
        Assert.True(d1 != d3);
    }

    [Fact]
    public void TestOperatorAdd()
    {
        long v1 = 2;
        long v2 = 1;
        Dimension d1 = v1;
        Dimension d2 = v2;
        var d3 = Dimension.Unknown;

        var d4 = d1 + d2;
        Assert.Equal(v1 + v2, d4);
        Assert.Equal(DimensionKind.Fixed, d4.Kind);

        d4 = d1 + d3;
        Assert.Equal(DimensionKind.Unknown, d4.Kind);

        d4 = d1 + v2;
        Assert.Equal(v1 + v2, d4);
        Assert.Equal(DimensionKind.Fixed, d4.Kind);
    }

    [Fact]
    public void TestOperatorSubtract()
    {
        long v1 = 2;
        long v2 = 1;
        Dimension d1 = v1;
        Dimension d2 = v2;
        var d3 = Dimension.Unknown;

        var d4 = d1 - d2;
        Assert.Equal(v1 - v2, d4);
        Assert.Equal(DimensionKind.Fixed, d4.Kind);

        d4 = d1 - d3;
        Assert.Equal(DimensionKind.Unknown, d4.Kind);
    }

    [Fact]
    public void TestOperatorMul()
    {
        long v1 = 2;
        long v2 = 1;
        Dimension d1 = v1;
        Dimension d2 = v2;
        var d3 = Dimension.Unknown;

        var d4 = d1 * d2;
        Assert.Equal(v1 * v2, d4);
        Assert.Equal(DimensionKind.Fixed, d4.Kind);

        d4 = d1 * d3;
        Assert.Equal(DimensionKind.Unknown, d4.Kind);
    }

    [Fact]
    public void TestOperatorDiv()
    {
        long v1 = 2;
        long v2 = 1;
        Dimension d1 = v1;
        Dimension d2 = v2;
        var d3 = Dimension.Unknown;

        var d4 = d1 / d2;
        Assert.Equal(v1 / v2, d4);
        Assert.Equal(DimensionKind.Fixed, d4.Kind);

        d4 = d1 / d3;
        Assert.Equal(DimensionKind.Unknown, d4.Kind);
    }

    [Fact]
    public void TestDimensionSum()
    {
        var dv = new DimVar("x");
        var negdv = -dv;
        var zero = dv + negdv;
        Assert.Equal(zero, new DimConst(0));
        var padding = new IR.Shapes.Padding(0, 128 - dv);
        var paded = dv + padding.Sum();
        Assert.Equal(paded, new DimConst(128));
        var paded2 = padding.Sum() + dv;
        Assert.Equal(paded2, new DimConst(128));
    }
}
