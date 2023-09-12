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
public class UnitTestFoldGetItemShapeOf : TransformTestBase
{
    [Fact]
    public void TestFoldGetItemShapeOf()
    {
        var input = new Var(new TensorType(DataTypes.Float32, new[] { 1, 3, Dimension.Unknown, 24 }));
        var data = Testing.Rand<float>(1, 3, 24, 24);
        var dict = new Dictionary<Var, IValue> { { input, Value.FromTensor(data) } };
        TestMatched<FoldGetItemShapeOf>(ShapeOf(input)[1], dict);
    }

    [Fact]
    public void TestFoldGetItemShapeOfWithCast()
    {
        var input = new Var(new TensorType(DataTypes.Float32, new[] { 1, 3, Dimension.Unknown, 24 }));
        var data = Testing.Rand<float>(1, 3, 24, 24);
        var dict = new Dictionary<Var, IValue> { { input, Value.FromTensor(data) } };
        TestMatched<FoldGetItemShapeOf>(Cast(ShapeOf(input), DataTypes.Int32)[1], dict);
    }

    [Fact]
    public void TestFoldGetItemShapeOfWithDynamic()
    {
        var input = new Var(new TensorType(DataTypes.Int32, new[] { 1, 3, Dimension.Unknown, 24 }));
        TestNotMatch<FoldGetItemShapeOf>(ShapeOf(input)[2]);
    }
}
