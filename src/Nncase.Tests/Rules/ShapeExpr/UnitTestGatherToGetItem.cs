// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes.Rules.ShapeExpr;
using Xunit;
using Xunit.Abstractions;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules.ShapeExpr;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public class UnitTestGatherToGetItem : TransformTestBase
{
    [Fact]
    public void TestGatherToGetItem()
    {
        var input = new[] { 1, 2, 3, 4 };
        var gather = Gather(input, 0, 0);
        TestMatched<GatherToGetItem>(gather);
    }

    [Fact]
    public void TestIndexNotScalar()
    {
        var input = new[] { 1, 2, 3, 4 };
        var gather = Gather(input, 0, new[] { 1, 2 });
        TestNotMatch<GatherToGetItem>(gather);
    }
}
