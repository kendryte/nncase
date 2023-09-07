// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeExpr;
using Xunit;
using Xunit.Abstractions;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules.ShapeExpr;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public class UnitTestSliceToGetItem : TransformTestBase
{
    [Fact]
    public void TestSliceToGetItem()
    {
        var input = new[] { 1, 2, 3, 4 };
        var gather = Squeeze(Slice(input, new[] { 1 }, new[] { 2 }, 1), new[]{0});
        TestMatched<SliceToGetItem>(gather);
    }

    [Fact]
    public void TestTooLong()
    {
        var input = new[] { 1, 2, 3, 4 };
        var gather = Slice(input, new[] { 1 }, new[] { 3 }, 1);
        TestNotMatch<SliceToGetItem>(gather);
    }
}
