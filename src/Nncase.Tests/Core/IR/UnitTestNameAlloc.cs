// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using Nncase;
using Nncase.IR;
using Nncase.Tests.TestFixture;
using Xunit;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Tests.CoreTest;

public sealed class UnitNameAlloc
{
    [Fact]
    public void TestGetUniqueVarName()
    {
        var expected1 = "x";
        var actual1 = NameAlloc.GetUniqueVarName(expected1);
        Assert.Equal(expected1, actual1);

        var expected2 = "x1";
        var actual2 = NameAlloc.GetUniqueVarName(expected1);
        Assert.Equal(expected2, actual2);

        var expected3 = "x2";
        var actual3 = NameAlloc.GetUniqueVarName(expected1);
        Assert.Equal(expected3, actual3);
    }

    [Fact]
    public void TestAddName()
    {
        var name = "y";
        var expected = "y1";
        NameAlloc.AddName(name);
        var actual = NameAlloc.GetUniqueVarName(name);
        Assert.Equal(expected, actual);
    }
}
