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
        Assert.Empty(v);
    }

    [Fact]
    public void TestNonVoid()
    {
        Const c1 = 1F;
        Const c2 = 2F;
        var tc = new TupleConst(new Const[] { c1, c2 });
        Assert.Equal(2, tc.Count);
        Assert.Equal(c1, tc[0]);

        var t = (ITuple)tc;
        Assert.Equal(2, t.Fields.Count);

        var list = (IReadOnlyList<Expr>)tc;
        Assert.Equal(c1, list[0]);
    }
}
