// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestBfloat16
{
    [Fact]
    public void TestInfinity()
    {
        var positiveInfinity = (BFloat16)1F / (BFloat16)0F;
        Assert.Equal(positiveInfinity, BFloat16.Infinity);
    }

    [Fact]
    public void TestNegInfinity()
    {
        var negInfinity = (BFloat16)(-1F) / (BFloat16)0F;
        Assert.Equal(negInfinity, BFloat16.NegInfinity);
    }

    [Fact]
    public void TestNan()
    {
        var nan = (BFloat16)(0F / 0F);
        Assert.Equal(nan, BFloat16.NaN);
    }

    [Fact]
    public void TestCompare()
    {
        var f = 1.234F;
        var a = (BFloat16)1.23F;
        var b = (BFloat16)1.23F;
        var c = (BFloat16)2.34F;

        Assert.True(a == b);
        Assert.True(a != c);
        Assert.True(a < c);
        Assert.True(a <= c);
        Assert.True(c > b);
        Assert.True(c >= b);

        Assert.Equal(a, b);
        Assert.NotEqual(a, c);
        Assert.NotEqual(a, f);
        Assert.True(a.Equals((object)b));
    }

    [Fact]
    public void TestGetHashCode()
    {
        ushort a = 0x1234;
        var b = BFloat16.FromRaw(a);
        Assert.True(a.GetHashCode() == b.GetHashCode());
    }

    [Fact]
    public void TestToString()
    {
        var a = (BFloat16)1.23F;
        Assert.True(a.ToString() == ((float)a).ToString());
    }
}
