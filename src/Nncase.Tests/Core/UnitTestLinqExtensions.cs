// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase;
using Nncase.IR;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestLinqExtensions
{
    [Fact]
    public void TestLinqExtensions()
    {
        var items = new[] { 1, 2, 3, 4 };
        var expected = new[] { 1, 2, 3, 4 };
        var result = items.TakeOrDefault(4, 0);
        Assert.Equal(expected, result);

        items = new[] { 1, 2, 3 };
        expected = new[] { 1, 2, 3, 0 };
        result = items.TakeOrDefault(4, 0);
        Assert.Equal(expected, result);

        items = new[] { 1, 2, 3 };
        expected = new[] { 1, 2, 3, 4, 4 };
        result = items.TakeOrDefault(5, 4);
        Assert.Equal(expected, result);

        items = Array.Empty<int>();
        expected = new[] { 0, 0, 0 };
        result = items.TakeOrDefault(3, 0);
        Assert.Equal(expected, result);
    }
}
