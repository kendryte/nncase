// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes;
using Nncase.PatternMatch;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.CoreTest;

public class UnitTestExprPattern
{
    [Fact]
    public void TestPatternNotEqual()
    {
        var p1 = IsWildcard();
        int p1hash = p1.GetHashCode();
        var p2 = IsBinary(b => true, IsConst(), IsConst());

        ExprPattern p3 = p1 with { TypePattern = IsScalar() };
        int p3hash = p3.GetHashCode();
        Assert.NotEqual(p1hash, p3hash);
    }

    [Fact]
    public void TestVarPattern()
    {
        var e = new Var("x", AnyType.Default);
        Assert.True(e.InferenceType());
        Pattern ep = e;
        Assert.IsType<VarPattern>(ep);
        Assert.True(CompilerServices.TryMatchRoot(e, ep, out _));
    }

    [Fact]
    public void TestTensorConstantPattern()
    {
        var con = (Const)1.1f;
        Assert.True(con.InferenceType());
        Pattern cp1 = con;
        Assert.IsType<TensorConstPattern>(cp1);

        var cp2 = IsConst((float x) => x > 1.2f);
        var cp3 = IsConst((int x) => x > 1);
        var cp4 = (TensorConstPattern)1.1f;

        Assert.True(CompilerServices.TryMatchRoot(con, cp1, out _));
        Assert.False(CompilerServices.TryMatchRoot(con, cp2, out _));
        Assert.False(CompilerServices.TryMatchRoot(con, cp3, out _));
        Assert.True(CompilerServices.TryMatchRoot(con, cp4, out _));
    }

    [Fact]
    public void TestTensorConstantPatternEqual()
    {
        var cp1 = (TensorConstPattern)1;
        var cp2 = (TensorConstPattern)1;
        Dictionary<TensorConstPattern, int> d = new();
        d.Add(cp1, 1);
        Assert.NotEqual(cp1, cp2);
        Assert.DoesNotContain(cp2, d.Keys);
        TensorConstPattern cp3 = IsTensorConst();
        TensorConstPattern cp4 = IsTensorConst();
        d.Add(cp3, 1);
        Assert.Equal(cp3, cp4);
        Assert.Contains(cp4, d.Keys);
    }

    [Fact]
    public void TestWildcardPattern()
    {
        var wc = IsWildcard();
        Assert.IsType<ExprPattern>(wc);
    }

    [Fact]
    public void TestWildCardPatternHash()
    {
        var wc = IsWildcard();
        var wc2 = new ExprPattern(x => true, null);
        var wc3 = new ExprPattern(x => true, null);
        var d = new Dictionary<ExprPattern, int>();
        d.Add(wc, 1);
        d.Add(wc2, 2);
        d.Add(wc3, 2);
    }

    [Fact]
    public void TestCallPattern()
    {
        var e = (Const)1 + Exp(10);
        Assert.True(e.InferenceType());
        var wc1 = IsWildcard();
        var wc2 = IsWildcard();
        var c = wc1 + wc2;
        Assert.IsType<CallPattern>(c);
        Assert.IsType<OpPattern<Binary>>(c.Target);
        Assert.IsType<ExprPattern>(c.Arguments[0]);
        Assert.IsType<ExprPattern>(c.Arguments[1]);

        CallPattern c2 = IsBinary(BinaryOp.Add, wc1, wc2);

        CallPattern c3 = IsBinary(x => x.BinaryOp is BinaryOp.Div or BinaryOp.Sub, wc1, wc2);

        Assert.True(CompilerServices.TryMatchRoot(e, c, out _));
        Assert.True(CompilerServices.TryMatchRoot(e, c2, out _));
        Assert.False(CompilerServices.TryMatchRoot(e, c3, out _));
    }

    [Fact]
    public void TestFunctionPattern()
    {
        var wc1 = IsWildcard();
        var wc2 = IsWildcard();
        var c = wc1 + wc2;
        var fp = new FunctionPattern(c, new[] { wc1, wc2 }, null);
        Assert.IsType<FunctionPattern>(fp);
        Assert.IsType<ExprPattern>(fp.Parameters[0]);
        Assert.IsType<ExprPattern>(fp.Parameters[1]);
        Assert.IsType<CallPattern>(fp.Body);
        Assert.IsType<ExprPattern>(((CallPattern)fp.Body).Arguments[0]);
        Assert.IsType<ExprPattern>(((CallPattern)fp.Body).Arguments[1]);
        _ = new FunctionPattern(c, IsVArgs(new[] { wc1, wc2 }), null);
        Assert.IsType<ExprPattern>(fp.Parameters[0]);
        Assert.IsType<ExprPattern>(fp.Parameters[1]);
    }

    [Fact]
    public void TestTuplePattern()
    {
        var wc1 = IsWildcard();
        var wc2 = IsWildcard();
        var t = PatternMatch.Utility.IsTuple(null, new[] { wc1, wc2 });
        Assert.IsType<TuplePattern>(t);
        Assert.IsType<ExprPattern>(t.Fields[0]);
        Assert.IsType<ExprPattern>(t.Fields[1]);

        var t2 = PatternMatch.Utility.IsTuple(IsVArgs(new[] { wc1, wc2 }));
        Assert.IsType<TuplePattern>(t2);
        Assert.IsType<ExprPattern>(t2.Fields[0]);
        Assert.IsType<ExprPattern>(t2.Fields[1]);
    }

    [Fact]
    public void TestVArgsPattern()
    {
        // var wc = IsWildcard();
        // var vwcs = new List<ExprPattern>();
        // var pattern = IsVArgsRepeat((n, param) =>
        // {
        //     for (int i = 0; i < n; i++)
        //     {
        //         var wc = IsWildcard();
        //         param.Add(wc);
        //         vwcs.Add(wc);
        //     }
        // },
        // (match, param) =>
        // {
        //     if (match == false)
        //     {
        //         param.Clear();
        //         vwcs.Clear();
        //     }
        // }
        // );

        // var tuple = new IR.Tuple(1, new IR.Tuple(6, 7, 8), 3, 4);
        // tuple.InferenceType();
        // Assert.True(pattern.MatchLeaf(tuple.Fields));
        // Assert.True(pattern[0].MatchLeaf(tuple[0]));
        // Assert.True(pattern[1].MatchLeaf(tuple[1]));
        // Assert.True(pattern[2].MatchLeaf(tuple[2]));
        // Assert.True(pattern[3].MatchLeaf(tuple[3]));
    }

    [Fact]
    public void TestVArgsPatternFunc()
    {
        var pat = IsTuple(IsVArgsRepeat(() => IsConst()));
        var expr1 = new IR.Tuple(1, 2, 3, 4, 5, 6);
        var expr2 = new IR.Tuple(new Var("x"), 2, 3, 4, 5, 6);

        Assert.True(CompilerServices.TryMatchRoot(expr1, pat, out _));
        Assert.Equal(pat.Fields.Count, expr1.Fields.Length);
        Assert.False(CompilerServices.TryMatchRoot(expr2, pat, out _));
        Assert.Equal(pat.Fields.Count, expr2.Fields.Length);
    }

    [Fact]
    public void TestAltPattern()
    {
        var lhs = IsWildcard();
        var rhs = IsWildcard();
        var is_op_call = IsCall(IsWildcard(), new[] { lhs, rhs });
        var x = (Const)1;
        var y = (Const)2;
        var z1 = x + y;
        var z2 = x * y;
        z1.InferenceType();
        z2.InferenceType();
        Assert.True(CompilerServices.TryMatchRoot(z1, is_op_call, out _));
        Assert.True(CompilerServices.TryMatchRoot(z2, is_op_call, out _));

        var is_op_call2 = IsCall(IsWildcard(), IsVArgs(new[] { lhs, rhs }));

        Assert.IsType<ExprPattern>(is_op_call2.Arguments[0]);
        Assert.IsType<ExprPattern>(is_op_call2.Arguments[1]);
    }

    [Fact]
    public void TestTypePattern()
    {
        var ttype = new TensorType(DataTypes.Float32, new[] { 10, 10 });
        var ty_pat = IsType(ttype);
        Assert.IsType<TypePattern>(ty_pat);
        Assert.True(ty_pat.MatchLeaf(ttype));
    }

    [Fact]
    public void TestDataTypePattern()
    {
        var ttype1 = new TensorType(DataTypes.Float32, new[] { 10, 10 });
        var ttype2 = new TensorType(DataTypes.Int16, new[] { 10 });
        var ty_pat = HasDataType(DataTypes.Float32);
        Assert.IsType<TypePattern>(ty_pat);
        Assert.True(ty_pat.MatchLeaf(ttype1));
        Assert.False(ty_pat.MatchLeaf(ttype2));
    }

    [Fact]
    public void TestShapePattern()
    {
        var shape = new int[] { 10, 10 };
        var sp = HasShape(shape);
        var ttype1 = new TensorType(DataTypes.Float32, new[] { 10, 10 });
        var ttype2 = new TensorType(DataTypes.Int16, new[] { 10 });
        Assert.True(sp.MatchLeaf(ttype1));
        Assert.False(sp.MatchLeaf(ttype2));
    }

    [Fact]
    public void TestPatternClone()
    {
        var pat = IsWildcard();
        var pat2 = IsWildcard();
        Assert.NotEqual(pat, pat2, ReferenceEqualityComparer.Instance);
    }

    [Fact]
    public void TestBuildExprFromPattern()
    {
        ConstPattern c0 = IsConst();
        _ = IsConst();
        var x = IsWildcard();
        _ = x + c0;
        _ = x - c0;
        _ = Equal(c0, 0);
    }
}
