// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualBasic.CompilerServices;
using NetFabric.Hyperlinq;
using Nncase;
using Nncase.Collections;
using Nncase.IR;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestWeakList
{
    [Fact]
    public void TestWeakList()
    {
        var weakList = new WeakList<object>();
        Assert.Equal(0, weakList.WeakCount);
        Assert.Throws<ArgumentOutOfRangeException>(() => weakList.GetWeakReference(-1));
        weakList.Add(1);
        weakList.Add(2);
        weakList.Add(3);
        weakList.Add(4);
        Assert.Equal(new WeakReference<object>(1).TryGetTarget(out _), weakList.GetWeakReference(0).TryGetTarget(out _));

#if false
        var list = new WeakList<object>();
        var obj1 = new object();
        var obj2 = new object();
        var obj3 = new object();

        // Add some objects to the list
        list.Add(obj1);
        list.Add(obj2);
        list.Add(obj3);

        // Check that GetWeakReference returns the expected WeakReference objects
        Assert.Equal(new WeakReference<object>(obj1).TryGetTarget(out _), list.GetWeakReference(0).TryGetTarget(out _));
        Assert.Equal(new WeakReference<object>(obj2).TryGetTarget(out _), list.GetWeakReference(1).TryGetTarget(out _));
        Assert.Equal(new WeakReference<object>(obj3).TryGetTarget(out _), list.GetWeakReference(2).TryGetTarget(out _));

        // Let some objects get collected
        obj1 = null;
        GC.Collect();

        // Check that the enumerator only returns the live objects
        Assert.Equal(new[] { obj2, obj3 }, list.ToList());

        // Add some more objects to the list
        var obj4 = new object();
        var obj5 = new object();
        list.Add(obj4);
        list.Add(obj5);

        // Let some more objects get collected
        obj3 = null;
        obj5 = null;
        GC.Collect();

        // Check that the enumerator only returns the live objects
        Assert.Equal(new[] { obj2, obj4 }, list.ToList());
#endif
    }

    [Fact]
    public void TestGetEnumerator()
    {
        // Arrange
        var list = new WeakList<object>();
        for (int i = 0; i < 100; i++)
        {
            list.Add(i);
        }

        var sizeBefore = list.ToList().Count;

        // Act
        _ = list.GetEnumerator();
        var sizeAfter = list.ToList().Count;

        // Assert
        Assert.Equal(sizeAfter, sizeBefore);
        Assert.True(sizeAfter > sizeBefore / 4);
    }
}
