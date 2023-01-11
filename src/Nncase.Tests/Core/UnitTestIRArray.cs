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
        Assert.True(a.Count == 0);
        Assert.True(a.IsReadOnly == true);
    }

    [Fact]
    public void TestSpan()
    {
        var array = new IR.IRArray<int>(Enumerable.Range(1, 100));
        Assert.True(array[0] == 1 && array[99] == 100);
        Assert.True(array[new Range(1, 3)].ToArray().Length == 2);
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
        Assert.True(a.Contains(1) == true);
        Assert.False(a.Contains(1000) == true);
    }

    [Fact]
    public void TestIndexOf()
    {
        var a = new IR.IRArray<int>(Enumerable.Range(1, 100));
        Assert.True(a.IndexOf(1) == 0);
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
