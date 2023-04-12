// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestUtf8Char
{
    [Fact]
    public void TestConvert()
    {
        byte b1 = 1;
        var u = (Utf8Char)(byte)b1;
        var b2 = (byte)u;
        Assert.Equal(b1, b2);
    }

    [Fact]
    public void TestEqual()
    {
        var u1 = (Utf8Char)(byte)1;
        var u2 = (Utf8Char)(byte)1;
        Assert.True(u1 == u2);
    }

    [Fact]
    public void TestNotEqual()
    {
        var u1 = (Utf8Char)(byte)1;
        var u2 = (Utf8Char)(byte)2;
        Assert.True(u1 != u2);
    }

    [Fact]
    public void TestLessThan()
    {
        var u1 = (Utf8Char)(byte)1;
        var u2 = (Utf8Char)(byte)2;
        Assert.True(u1 < u2);
    }

    [Fact]
    public void TestLessOrEqual()
    {
        byte b = 1;
        var u1 = (Utf8Char)b;
        var u2 = (Utf8Char)b;
        var u3 = (Utf8Char)(byte)2;
        Assert.True(u1 <= u2);
        Assert.True(u2 <= u3);
        Assert.False(u3 <= u1);
    }

    [Fact]
    public void TestGreaterThan()
    {
        var u1 = (Utf8Char)(byte)1;
        var u2 = (Utf8Char)(byte)2;
        Assert.False(u1 > u2);
        Assert.True(u2 > u1);
    }

    [Fact]
    public void TestGreaterOrEqual()
    {
        byte b = 1;
        var u1 = (Utf8Char)b;
        var u2 = (Utf8Char)b;
        var u3 = (Utf8Char)(byte)2;
        Assert.True(u1 >= u2);
        Assert.False(u2 >= u3);
        Assert.True(u3 >= u1);
    }

    [Fact]
    public void TestEquals1()
    {
        byte b = 1;
        var u1 = (Utf8Char)b;
        var u2 = (Utf8Char)b;
        Assert.Equal(u1, u2);
    }

    [Fact]
    public void TestNotEquals2()
    {
        byte b = 1;
        var u1 = (Utf8Char)b;
        var u2 = (Utf8Char)b;
        var u3 = (Utf8Char)(byte)2;
        Assert.True(u1.Equals((object)u2));
        Assert.False(u1.Equals((object)u3));
        Assert.False(u1.Equals((object)b));
    }

    [Fact]
    public void TestGetHashCode()
    {
        byte b = 1;
        var u = (Utf8Char)b;
        Assert.Equal(b.GetHashCode(), u.GetHashCode());
    }
}
