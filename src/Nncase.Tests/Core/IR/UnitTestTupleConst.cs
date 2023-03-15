// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using Nncase;
using Nncase.IR;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestTupleConst
{
    [Fact]
    public void TestVoid()
    {
        var v = TupleConst.Void;
        Assert.Empty(v.Value);
    }

    [Fact]
    public void TestNonVoid()
    {
        var c1 = Value.FromConst(1F);
        var c2 = Value.FromConst(2F);
        var tc = new TupleConst(new TupleValue(new[] { c1, c2 }));
        Assert.Equal(2, tc.Count);
        Assert.Equal(c1, tc[0]);

        var t = (ITuple)tc;
        Assert.Equal(2, t.Count);

        var list = (IReadOnlyList<IValue>)tc.Value;
        Assert.Equal(c1, list[0]);
    }
}
