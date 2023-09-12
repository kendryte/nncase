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
public class UnitTestFoldSplitShapeOf : TransformTestBase
{
    [Fact]
    public void TestFoldSplitShapeOf()
    {
        var input = Testing.Rand<float>(1, 3, 24, 24);
        var shape = ShapeOf(input);
        var newShape = Stack(new IR.Tuple(shape[0], shape[1], shape[2], shape[3]), 0);
        TestMatched<FoldSplitShapeOf>(newShape);
    }
}
