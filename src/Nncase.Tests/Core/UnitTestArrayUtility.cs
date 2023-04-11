// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase;
using Nncase.IR;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestArrayUtility
{
    [Fact]
    public void TestToExprArray()
    {
        var a = new[] { 1, 2, 3, 4 };
        var b = new Expr[] { 1, 2, 3, 4 };
        Assert.Equal(b, ArrayUtility.ToExprArray(a));
    }

    [Fact]
    public void TestToConcat()
    {
        var a = new[] { 1, 2, 3, 4 };
        var b = new[] { 5, 6, 7, 8 };
        var result1 = new[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        Assert.Equal(result1, ArrayUtility.Concat<int>(a, b));

        var value1 = 1;
        var result2 = new[] { 1, 1, 2, 3, 4 };
        Assert.Equal(result2, ArrayUtility.Concat<int>(value1, a));
    }

    [Fact]
    public void TestTo2D()
    {
        int[,] result = { { 1, 2 }, { 3, 4 } };
        int[][] source = new int[2][];
        source[0] = new[] { 1, 2 };
        source[1] = new[] { 3, 4 };
        Assert.Equal(result, ArrayUtility.To2D<int>(source));
        source[1] = new[] { 3, 4, 5 };
        Assert.Throws<InvalidOperationException>(() => ArrayUtility.To2D(source));
    }
}
