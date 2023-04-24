// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualBasic.CompilerServices;
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
    }
}
