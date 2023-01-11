// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestPointer
{
    [Fact]
    public unsafe void TestConstructor()
    {
        var value = 2023;
        var addr = (ulong)&value;
        var p = new Pointer<int>(addr);
        Assert.True(p.Value == addr);
        Assert.True(Pointer<int>.ElemSize == sizeof(int));
    }

    [Fact]
    public unsafe void TestCompare()
    {
        var f1 = 1.234F;
        var f2 = 1.567F;

        var a = new Pointer<float>((ulong)&f1);
        var b = new Pointer<float>((ulong)&f1);
        var c = new Pointer<float>((ulong)&f2);

        Assert.True(a == b);
        Assert.True(a != c);
        Assert.True(b != c);

        Assert.True(a.Equals(b));
        Assert.False(a.Equals(c));
        Assert.False(a.Equals(f1));
        Assert.True(a.Equals((object)b));
    }

    [Fact]
    public unsafe void TestGetHashCode()
    {
        var f = 1.234F;
        var addr = (ulong)&f;
        var p = new Pointer<float>(addr);
        Assert.True(p.GetHashCode() == addr.GetHashCode());
        Assert.False(p.GetHashCode() == f.GetHashCode());
    }
}
