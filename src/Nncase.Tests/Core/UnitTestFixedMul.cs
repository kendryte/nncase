// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase;
using Nncase.Tests.TestFixture;
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
        Assert.Equal(mul, a.Mul);
        Assert.Equal(shift, a.Shift);
    }

    [Fact]
    public void TestRoundedMul()
    {
        var mul = 2.34F;
        var shift1 = (sbyte)0;
        var a = new FixedMul(mul, shift1);
        Assert.Equal(System.MathF.Round(mul), a.RoundedMul);
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
        Assert.Equal(a, b);
        Assert.NotEqual(a, c);
        Assert.True(a.Equals((object)b));
        Assert.False(a.Equals((object)c));
        Assert.False(a.Equals(mul));
    }

    [Fact]
    public void TestGetHashCode()
    {
        var mul = 2.0F;
        var shift = (sbyte)0;
        var a = new FixedMul(mul, shift);
        var b = new FixedMul(mul, shift);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }
}
