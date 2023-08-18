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
public class UnitTestFoldGetItem : TransformTestBase
{
    [Fact]
    public void TestFoldStackGetItem()
    {
        Expr input = new[] { 1, 2, 3, 4 };
        var s = Stack(new IR.Tuple(new[] { input[0], input[1], input[2], input[3] }), 0);
        TestMatched<FoldStackGetItem>(s);
    }

    [Fact]
    public void TestFoldStackGetItemDyn()
    {
        var input = Tensor.From(new[] { 1, 2, 3 });
        var inputVar = new Var(new TensorType(input.ElementType, input.Shape));
        var abs = IR.F.Math.Abs(inputVar);
        var s = Stack(new IR.Tuple(new[] { abs[0], abs[1], abs[2] }), 0);
        var body = new If(true, new[] { 3, 2, 1 }, s);
        TestMatched<FoldStackGetItem>(body, new Dictionary<Var, IValue> { { inputVar, Value.FromTensor(input) } });
    }

    [Fact]
    public void TestFoldSqueezeGetItem()
    {
        var shape = new Expr[] { 1, 80, 4, 1 };
        var s = Stack(new IR.Tuple(shape), 0);
        var result = Stack(new IR.Tuple(s[0], s[1], s[2]), 0);
        TestMatched<FoldStackGetItem>(result);
    }

    [Fact]
    public void TestFoldStackGetItemErrIndex()
    {
        Expr input = new[] { 1, 2, 3, 4 };
        var s = Stack(new IR.Tuple(new[] { input[0], input[1], input[3], input[2] }), 0);
        TestNotMatch<FoldStackGetItem>(s);
    }

    [Fact]
    public void TestFoldStackGetItemDiffInput()
    {
        Expr i0 = new[] { 1, 2, 3, 4 };
        Expr i1 = new[] { 5, 6, 7, 8 };
        Expr i2 = new[] { 9, 10, 11, 12 };
        var s = Stack(new IR.Tuple(new[] { i0, i1, i2 }), 0);
        TestNotMatch<FoldStackGetItem>(s);
    }
}
