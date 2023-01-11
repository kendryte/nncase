// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase;
using Nncase.TestFixture;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestFixedMul
{
    [Fact]
    public void TestConstructor()
    {
        var mul = 2.0F;
        var shift = (sbyte)0;
        var a = new FixedMul(mul, shift);
        Assert.True(a.Mul == mul);
        Assert.True(a.Shift == shift);
    }

    [Fact]
    public void TestRoundedMul()
    {
        var mul = 2.34F;
        var shift1 = (sbyte)0;
        var a = new FixedMul(mul, shift1);
        Assert.True(a.RoundedMul == System.MathF.Round(mul));
    }

    [Fact]
    public void TestCompare()
    {
        var mul = 2.0F;
        var shift1 = (sbyte)0;
        var shift2 = (sbyte)2;
        var a = new FixedMul(mul, shift1);
        var b = new FixedMul(mul, shift1);
        var c = new FixedMul(mul, shift2);

        Assert.True(a.Equals(b));
        Assert.False(a.Equals(c));
        Assert.True(a.Equals((object)b));
        Assert.False(a.Equals((object)c));
        Assert.False(a.Equals(mul));
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
