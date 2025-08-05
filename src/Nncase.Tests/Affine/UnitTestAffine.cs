﻿// Copyright (c) Canaan Inc. All rights reserved.
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
        var islmap = AffineUtility.AsMap(affmap);
        Assert.Equal("{ [d0, d1, d2, d3] -> [d3, d2, d0, d1] }", islmap.ToString());
    }

    [Fact]
    public void TestToSet()
    {
        {
            // simple case
            var x = new DimVar("x")
            {
                Metadata = new()
                {
                    Range = new(8, 16),
                },
            };
            var tile = 12;
            var dim = (x + tile - 1) / tile;
            using var ctx = Isl.ctx.Create();
            var set = dim.ToSet(out var dimVars);
            Assert.Equal(1, set.dim_min_val(0).num_si());
            Assert.Equal(2, set.dim_max_val(0).num_si());
        }

        {
            // one dimension with multiple variables
            var x = new DimVar("x")
            {
                Metadata = new()
                {
                    Range = new(8, 16),
                },
            };
            var y = new DimVar("y")
            {
                Metadata = new()
                {
                    Range = new(4, 8),
                },
            };
            var tile = 12;
            var dims = new[] { (x + tile - 1) / tile, x + y - 4 };
            using var ctx = Isl.ctx.Create();
            var set = ISLUtility.ToSet(dims, out var dimVars);
            Assert.Equal(1, set.dim_min_val(0).num_si());
            Assert.Equal(2, set.dim_max_val(0).num_si());
            Assert.Equal(8, set.dim_min_val(1).num_si());
            Assert.Equal(20, set.dim_max_val(1).num_si());
        }
    }

    [Fact]
    public void TestToDomain()
    {
        {
            // simple case
            var x = new DimVar("x")
            {
                Metadata = new()
                {
                    Range = new(8, 16),
                },
            };

            var shape = new RankedShape(x, 16, 16);
            using var ctx = Isl.ctx.Create();
            var domain = ISLUtility.ToDomain(shape, out _);
            Assert.False(domain.dim_max(0).is_cst());
            Assert.Equal(15, domain.dim_max_val(0).num_si());
        }

        {
            // divide case
            var x = new DimVar("x")
            {
                Metadata = new()
                {
                    Range = new(8, 16),
                },
            };

            var shape = new RankedShape((x + 11) / 12, 16 / 8);
            using var ctx = Isl.ctx.Create();
            var domain = ISLUtility.ToDomain(shape, out _);
            Assert.False(domain.dim_max(0).is_cst());
            Assert.True(domain.dim_max(1).is_cst());
            Assert.Equal(1, domain.dim_max_val(0).num_si());
            var domain2 = ISLUtility.ToParametricDomain(shape, out _);
            Assert.False(domain2.dim_max(0).is_cst());
            Assert.True(domain2.dim_max(1).is_cst());
            Assert.Equal(1, domain2.dim_max_val(0).num_si());
        }

        {
            // divide case 2
            var x = new DimVar("x")
            {
                Metadata = new()
                {
                    Range = new(1, 128),
                },
            };

            var shape = new RankedShape((x + 8 - 1) / 8, 64);
            using var ctx = Isl.ctx.Create();
            var domain = ISLUtility.ToDomain(shape, out _);
            Assert.False(domain.dim_max(0).is_cst());
            Assert.True(domain.dim_max(1).is_cst());
            Assert.Equal(15, domain.dim_max_val(0).num_si());
        }

        {
            // divide case 3
            var x = new DimVar("x")
            {
                Metadata = new()
                {
                    Range = new(1, 128),
                },
            };
            var shape = new RankedShape(x / 128, 1);
            using var ctx = Isl.ctx.Create();
            var domain = ISLUtility.ToDomain(shape, out _);
            Assert.False(domain.dim_max(0).is_cst());
            Assert.Equal(0, domain.dim_max(0).max_val().num_si());
            Assert.Equal(-1, domain.dim_max(0).min_val().num_si());
        }
    }

    [Fact]
    public void TestToDimension()
    {
        using var ctx = Isl.ctx.Create();
        {
            var map = new Isl.map(ctx, "{ [a,b] -> [a+b] }");
            var build = Isl.ast_build.from_context(new Isl.set(ctx, "{ [a,b]:}"));
            var access = build.access_from(map.as_pw_multi_aff());
            var expr = ISLUtility.ToDimension(access.op_arg(1), new Dictionary<string, Dimension>() {
                { "a", new DimVar("a") },
                { "b", new DimVar("b") },
            });
            var sum = Assert.IsType<DimSum>(expr);
            Assert.IsType<DimVar>(sum.Operands[0]);
            Assert.IsType<DimVar>(sum.Operands[1]);
        }

        {
            var a = new Isl.pw_aff(ctx, "{ [n] -> [2] : n = 1 }");
            var b = new Isl.pw_aff(ctx, "{ [n] -> [3] : n = 2 }");
            var c = a.union_add(b);
            var build = Isl.ast_build.from_context(new Isl.set(ctx, "{ [n]: }"));
            var astExpr = build.expr_from(c);
            var expr = ISLUtility.ToDimension(astExpr, new Dictionary<string, Dimension>() {
                { "n", new DimVar("n") },
            });
            var select = Assert.IsType<DimCompareAndSelect>(expr);
            Assert.Equal(CompareOp.Equal, select.CompareOp);
            Assert.IsType<DimVar>(select.Value);
            Assert.IsType<DimConst>(select.Expected);
            Assert.IsType<DimConst>(select.TrueValue);
            Assert.IsType<DimConst>(select.FalseValue);
        }

        {
            var a = new Isl.pw_aff(ctx, "{ [n] -> [2] : n <= 2; [n] -> [n - 2] : n > 2 }");
            var build = Isl.ast_build.from_context(new Isl.set(ctx, "{ [n]: }"));
            var astExpr = build.expr_from(a);
            var expr = ISLUtility.ToDimension(astExpr, new Dictionary<string, Dimension>() {
                { "n", new DimVar("n") },
            });
            var select = Assert.IsType<DimCompareAndSelect>(expr);
            Assert.Equal(CompareOp.LowerOrEqual, select.CompareOp);
            Assert.IsType<DimVar>(select.Value);
            Assert.Equal(2, select.Expected);
            Assert.Equal(2, select.TrueValue);
            Assert.IsType<DimSum>(select.FalseValue);
        }

        {
            var a = new Isl.pw_aff(ctx, "{ [n] -> [2] : n >= 2; [n] -> [n - 2] : n < 2 }");
            var build = Isl.ast_build.from_context(new Isl.set(ctx, "{ [n]: }"));
            var astExpr = build.expr_from(a);
            var expr = ISLUtility.ToDimension(astExpr, new Dictionary<string, Dimension>() {
                { "n", new DimVar("n") },
            });
            var select = Assert.IsType<DimCompareAndSelect>(expr);
            Assert.Equal(CompareOp.GreaterOrEqual, select.CompareOp);
            Assert.IsType<DimVar>(select.Value);
            Assert.Equal(2, select.Expected);
            Assert.Equal(2, select.TrueValue);
            Assert.IsType<DimSum>(select.FalseValue);
        }

        {
            var a = new Isl.pw_aff(ctx, "[N, ao, bo, co] -> { [(32)] : N <= 128 and ao >= 0 and 32ao <= -32 + N and 0 <= bo <= 3 and 0 <= co <= 3; [(N - 32ao)] : N <= 128 and ao >= 0 and -31 + N <= 32ao < N and 0 <= bo <= 3 and 0 <= co <= 3 }");
            var build = new Isl.ast_build(ctx);
            var astExpr = build.expr_from(a);
            var expr = ISLUtility.ToDimension(astExpr, new Dictionary<string, Dimension>() {
                { "N", new DimVar("N") },
                { "ao", new DimVar("ao") },
            });
            var select = Assert.IsType<DimCompareAndSelect>(expr);
            Assert.Equal(CompareOp.GreaterOrEqual, select.CompareOp);
        }

        {
            ctx.set_ast_build_detect_min_max(1);
            var a = new Isl.pw_aff(ctx, "[N, ao, bo, co] -> { [(32)] : N <= 128 and ao >= 0 and 32ao <= -32 + N and 0 <= bo <= 3 and 0 <= co <= 3; [(N - 32ao)] : N <= 128 and ao >= 0 and -31 + N <= 32ao < N and 0 <= bo <= 3 and 0 <= co <= 3 }");
            var build = new Isl.ast_build(ctx);
            var astExpr = build.expr_from(a);
            var expr = ISLUtility.ToDimension(astExpr, new Dictionary<string, Dimension>() {
                { "N", new DimVar("N") },
                { "ao", new DimVar("ao") },
            });
            Assert.IsType<DimMin>(expr);
        }
    }

    [Fact]
    public void TestRoundTrip()
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
        var dims = new Dimension[] { d0_Op0_L2 + d0_Op0_L1 + (-1 * d0_Op0_L2) + (-1 * d0_Op0_L1), d1_Op0_L2 + d1_Op0_L1 + (-1 * d1_Op0_L2) + (-1 * d1_Op0_L1), d2_Op0_L2 + d2_Op0_L1 + (-1 * d2_Op0_L2) + (-1 * d2_Op0_L1), d3_Op0_L2 + d3_Op0_L1 + (-1 * d3_Op0_L2) + (-1 * d3_Op0_L1) };
        var newDims = ISLUtility.RoundTrip(dims);
        var n = Assert.IsType<DimConst>(newDims[0]);
        Assert.Equal(0, n.Value);
        var c = Assert.IsType<DimConst>(newDims[1]);
        Assert.Equal(0, c.Value);
        var h = Assert.IsType<DimConst>(newDims[2]);
        Assert.Equal(0, h.Value);
        var w = Assert.IsType<DimConst>(newDims[3]);
        Assert.Equal(0, w.Value);
    }
}
