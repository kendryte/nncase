// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestIRArray
{
    [Fact]
    public void TestConstructor()
    {
        var a = new IR.IRArray<float>();
        Assert.True(a.IsDefaultOrEmpty);
        Assert.Empty(a);
        Assert.True(a.IsReadOnly);
    }

    [Fact]
    public void TestSpan()
    {
        var array = new IR.IRArray<int>(Enumerable.Range(1, 100));
        Assert.Equal(1, array[0]);
        Assert.Equal(100, array[99]);
        Assert.Equal(2, array[new Range(1, 3)].ToArray().Length);
    }

    [Fact]
    public void Testcompare()
    {
        var a = new IR.IRArray<int>(Enumerable.Range(1, 100));
        var b = new IR.IRArray<int>(Enumerable.Range(1, 100));
        var c = new IR.IRArray<int>(Enumerable.Range(2, 101));

        Assert.True(a == b);
        Assert.True(a != c);
        Assert.True(a.Equals((object)b));
        Assert.False(b.Equals((object)c));
    }

    [Fact]
    public void TestContains()
    {
        var a = new IR.IRArray<int>(Enumerable.Range(1, 100));
        Assert.Contains(1, a);
        Assert.DoesNotContain(1000, a);
    }

    [Fact]
    public void TestIndexOf()
    {
        var a = new IR.IRArray<int>(Enumerable.Range(1, 100));
        Assert.Equal(0, a.IndexOf(1));
    }

    [Fact]
    public void TestException()
    {
        var a = new IR.IRArray<int>(Enumerable.Range(1, 100));
        Assert.Throws<InvalidOperationException>(() => a.Add(101));
        Assert.Throws<InvalidOperationException>(() => a.Clear());
        Assert.Throws<InvalidOperationException>(() => a.Insert(0, 101));
        Assert.Throws<InvalidOperationException>(() => a.Remove(1));
        Assert.Throws<InvalidOperationException>(() => a.RemoveAt(0));
    }
}
