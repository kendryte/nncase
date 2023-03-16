// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Extensions.Hosting;
using Nncase.IR;
using Nncase.Passes;
using Nncase.PatternMatch;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.MatchTest;

public sealed class UnitTestEGraphMatch
{
    [Fact]
    public void TestWildCardRecursion()
    {
        var wcx = IsWildcard("x");
        var wcy = IsWildcard("y");
        var pat = wcx + (wcy + IsConst(1));
        Var x = "x", y = "y";
        Expr expr = x + (y + 1);

        Assert.True(CompilerServices.TryEMatchRoot(expr, pat, out var res));
        Assert.IsType<Var>(res[0][wcx]);
        Assert.IsType<Var>(res[0][wcy]);
    }

    [Fact]
    public void TestWildCardRecursion2()
    {
        var wcx = IsWildcard("x");
        var wcy = IsWildcard("y");
        var pat = wcx + (wcy + IsConst(1));
        Var x = "x", y = "y";
        Expr expr = x + (y + 1);

        Assert.True(CompilerServices.TryEMatchRoot(expr, pat, out var res));
        Assert.Single(res);
    }

    [Fact]
    public void TestMatchOpAdd()
    {
        var wc1 = IsWildcard();
        var pat = wc1 + 1;

        var a = new Var("a");
        var wce1 = (a * 100) - ((Expr)32 / 320);
        var e = wce1 + 1;

        Assert.True(CompilerServices.TryEMatchRoot(e, pat, out var matchs));
        Assert.Single(matchs);
        var result = matchs[0];
        Assert.Equal(result[wc1], wce1);
    }

    [Fact]
    public void TestMatchOpOR()
    {
        var x = new Var("a");
        var y = x + 10;
        var y1 = y - 10;

        var px = IsWildcard();
        var py = IsBinary(op => op.BinaryOp is BinaryOp.Add or BinaryOp.Sub, px, IsConst(10));

        Assert.True(CompilerServices.TryEMatchRoot(y, py, out var matchs));
        Assert.Single(matchs);

        Assert.True(CompilerServices.TryEMatchRoot(y1, py, out var matchs2));
        Assert.Equal(2, matchs2.Count);

        var py1 = IsUnary(UnaryOp.Abs, px);
        Assert.False(CompilerServices.TryEMatchRoot(y1, py1, out var _));
    }

    [Fact]
    public void MatchFunction()
    {
        Var x = "x";
        Var y = "y";

        var wc1 = IsWildcard("x");
        var wc2 = IsWildcard("y");

        Expr func = new Function(x + y - 1200, x, y);

        var pat_1 = IsFunction("pt1", wc1 + wc2 - 1200, wc1, wc2);
        var pat_2 = IsFunction("pt1", wc1 - wc2, wc1, wc2);

        Assert.True(CompilerServices.TryEMatchRoot(func, pat_1, out var res_1));
        Assert.Single(res_1);
        Assert.False(CompilerServices.TryEMatchRoot(func, pat_2, out _));
    }

    [Fact]
    public void TestMatchVArgs()
    {
        _ = IsWildcard("x");

        var nest_tuple = new IR.Tuple(4, 5, 6);
        var tuple = new IR.Tuple(1, nest_tuple, 3);
        Expr expr = Concat(tuple, 0);
        CompilerServices.InferenceType(expr);

        var vpat = IsConcat(IsTuple("tp"), IsConst(0));

        Assert.True(CompilerServices.TryEMatchRoot(expr, vpat, out var eMatches));
        Assert.Single(eMatches);
    }

    [Fact]
    public void TestMatchVArgsTwice()
    {
        ConstPattern wcaxis = IsConst();

        var tuple_lhs = new IR.Tuple(1, new Var(), 3);
        var tuple_rhs = new IR.Tuple(4, 5, 6);
        Expr expr = Concat(tuple_lhs, 0) + Concat(tuple_rhs, 1);

        var vpat = IsConcat(IsTuple("tp"), wcaxis);

        Assert.True(CompilerServices.TryEMatchRoot(expr, vpat, out var eMatches));
        Assert.Equal(2, eMatches.Count);
    }

    [Fact]
    public void TestMatchVArgsRecursion()
    {
        Var x = "x";
        Const y = 4;
        Expr z = (Const)1 + 2;

        Const perm = 123;
        Expr expr = Concat(
            new IR.Tuple(
                Transpose(x, perm),
                Transpose(y, perm),
                Transpose(z, perm)),
            0);

        var wc = IsWildcard("wc");
        var wcperm = IsWildcard("perm");
        var wcaxis = IsWildcard("axis");

        var pattern = IsConcat(IsTuple(IsVArgsRepeat("wcvargs", () => IsTranspose(IsWildcard(), wcperm))), wcaxis);

        Assert.True(CompilerServices.TryEMatchRoot(expr, pattern, out var results));
        Assert.Single(results);
        var result = results[0];
        var wcvargs = (IReadOnlyList<Expr>)result["wcvargs"];
        Assert.Equal(((Call)wcvargs[0]).Arguments[0], x);
        Assert.Equal(((Call)wcvargs[1]).Arguments[0], y);
        Assert.Equal(((Call)wcvargs[2]).Arguments[0], z);
        Assert.Equal(result[wcperm], perm);
        Assert.Equal(result[wcaxis], (Const)0);
    }

    [Fact]
    public void TestMatchSameConstPatternTwice()
    {
        var x = (Const)1;
        Expr expr = (x * x) + 12 - x;
        var xpat = IsConst();
        Assert.True(CompilerServices.TryEMatchRoot(expr, IsBinary(op => true, xpat, xpat), out var result));
        Assert.Single(result);

        Assert.False(CompilerServices.TryEMatchRoot((x * 2) + 12 - x, IsBinary(op => true, xpat, xpat), out var result2));
    }

    [Fact]
    public void TestMatchSameWildCardPatternTwice()
    {
        var x = (Const)1;
        Expr expr = (x * x) + 12 - x;
        var xpat = IsWildcard();
        Assert.True(CompilerServices.TryEMatchRoot(expr, IsBinary(op => true, xpat, xpat), out var result));
        Assert.Single(result);

        Assert.False(CompilerServices.TryEMatchRoot((x * 2) + 12 - x, IsBinary(op => true, xpat, xpat), out var result2));
    }

    [Fact]
    public void TestMatchCallFusion()
    {
        Fusion fusion;
        {
            var fusion_input = new Var(new TensorType(DataTypes.Float32, new[] { 1, 2, 3, 4 }));
            fusion = new Fusion(Callable.StackVMModuleKind, IR.F.Tensors.Transpose(fusion_input, new[] { 0, 3, 1, 2 }), new[] { fusion_input });
        }

        var call = new Call(fusion, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 1, 2, 3, 4 }));

        var pattern = IsCall("callee", IsFusion("callee_fusion", Callable.StackVMModuleKind, IsWildcard(), IsVArgs(IsVar())), IsWildcard("callee_input"));

        Assert.True(CompilerServices.TryEMatchRoot(call, pattern, out var result));
        Assert.Single(result);
    }
}
