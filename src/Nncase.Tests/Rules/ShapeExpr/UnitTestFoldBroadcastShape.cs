// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.Passes.Rules.ShapeExpr;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.ShapeExpr;

namespace Nncase.Tests.Rules.ShapeExpr;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestFoldBroadcastShape : TransformTestBase
{
    [Fact]
    public void TestFoldBroadcastShape()
    {
        var b1 = BroadcastShape(new[] { (Expr)Tensor.From(new[] { 1, 3 }), Tensor.From(new[] { 1 }) });
        var b2 = BroadcastShape(new[] { (Expr)b1, Tensor.From(new[] { 1, 1 }) });
        TestMatched<FoldBroadcastShape>(b2);
    }

    [Fact]
    public void TestFoldBroadcastShapeConst()
    {
        var input = Testing.Rand<float>(1, 3, 1, 1);
        var b = BroadcastShape(new Expr[] { new[] { 1, 3 }, Array.Empty<int>(), IR.F.Tensors.ShapeOf(input) });
        TestMatched<FoldBroadcastShapeConst>(b);
    }
}
