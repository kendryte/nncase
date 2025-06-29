// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Reactive;
using Microsoft.Extensions.DependencyInjection;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Utilities;
using Xunit;
using Isl = IntegerSetLibrary;

namespace Nncase.Tests.AffineTest;

public class UnitTestAffine
{
    [Fact]
    public void TestAffineMapToIslMap()
    {
        using var ctx = Isl.ctx.Create();
        var affmap = AffineMap.Permutation([3, 2, 0, 1]);
        var islmap = AffineUtility.AsIslMap(affmap);
        Assert.Equal("{ [d0, d1, d2, d3] -> [d3, d2, d0, d1] }", islmap.ToString());
    }

    [Fact]
    public void TestDimensionToDomain()
    {
        var x = new DimVar("x")
        {
            Metadata = new()
            {
                Range = new(8, 16),
            },
        };

        var shape = new RankedShape(x, 16, 16);
        using var ctx = Isl.ctx.Create();
        var domain = ISLUtility.AsDomain(shape);
        Assert.False(domain.dim_max(0).is_cst());
        Assert.Equal(15, domain.dim_max_val(0).num_si());
    }

    [Fact]
    public void TestAstExprToDimension()
    {
        using var ctx = Isl.ctx.Create();
        var map = new Isl.map(ctx, "{ [a,b] -> [a+b] }");
        var build = Isl.ast_build.from_context(new Isl.set(ctx, "{ [a,b]:}"));
        var access = build.access_from(map.as_pw_multi_aff());
        var expr = ISLUtility.AsDimension(access.op_arg(1), new Dictionary<string, Dimension>() {
            { "a", new DimVar("a") },
            { "b", new DimVar("b") },
        });
        var sum = Assert.IsType<DimSum>(expr);
        Assert.IsType<DimVar>(sum.Operands[0]);
        Assert.IsType<DimVar>(sum.Operands[1]);
    }

    [Fact]
    public void TestSimplifyDimension()
    {
        using var ctx = Isl.ctx.Create();
        var d0_Op0_L2 = new DimVar("d0_Op0_L2");
        var d1_Op0_L2 = new DimVar("d1_Op0_L2");
        var d2_Op0_L2 = new DimVar("d2_Op0_L2");
        var d3_Op0_L2 = new DimVar("d3_Op0_L2");
        var d0_Op0_L1 = new DimVar("d0_Op0_L1");
        var d1_Op0_L1 = new DimVar("d1_Op0_L1");
        var d2_Op0_L1 = new DimVar("d2_Op0_L1");
        var d3_Op0_L1 = new DimVar("d3_Op0_L1");
        var shape = new RankedShape(d0_Op0_L2 + d0_Op0_L1 + (-1 * d0_Op0_L2) + (-1 * d0_Op0_L1), d1_Op0_L2 + d1_Op0_L1 + (-1 * d1_Op0_L2) + (-1 * d1_Op0_L1), d2_Op0_L2 + d2_Op0_L1 + (-1 * d2_Op0_L2) + (-1 * d2_Op0_L1), d3_Op0_L2 + d3_Op0_L1 + (-1 * d3_Op0_L2) + (-1 * d3_Op0_L1));
        var newShape = ISLUtility.Simplify(shape);
        var n = Assert.IsType<DimConst>(newShape[0]);
        Assert.Equal(0, n.Value);
        var c = Assert.IsType<DimConst>(newShape[1]);
        Assert.Equal(0, c.Value);
        var h = Assert.IsType<DimConst>(newShape[2]);
        Assert.Equal(0, h.Value);
        var w = Assert.IsType<DimConst>(newShape[3]);
        Assert.Equal(0, w.Value);
    }
}
