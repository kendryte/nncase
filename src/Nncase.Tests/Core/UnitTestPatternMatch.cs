// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Extensions.Hosting;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Random;
using Nncase.Passes;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;
using Nncase.TIR;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using Dimension = Nncase.IR.Dimension;
using Function = Nncase.IR.Function;
using Math = Nncase.PatternMatch.F.Math;
using Utility = Nncase.PatternMatch.Utility;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestPatternMatch
{
    [Fact]
    public void TestFunctionPattern()
    {
        Var x = "x";
        Var y = "y";

        var wc1 = IsWildcard("x");
        var wc2 = IsWildcard("y");

        Expr func = new Function(x + y - 1200, x, y);

        var pat1 = IsFunction(wc1 + wc2 - 1200, wc1, wc2);
        var pat2 = IsFunction(wc1 - wc2, wc1, wc2);

        Assert.True(CompilerServices.TryEMatchRoot(func, pat1, out var res1));
        Assert.Single(res1);
        Assert.False(CompilerServices.TryEMatchRoot(func, pat2, out _));
    }

    [Fact]
    public void TestPatternMatch()
    {
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Abs(1f), Math.Abs(1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Ceil(1f), Math.Ceil(1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Cos(1f), Math.Cos(1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Exp(1f), Math.Exp(1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Floor(1f), Math.Floor(1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Log(1f), Math.Log(1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Neg(1f), Math.Neg(1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Round(1f), Math.Round(1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Rsqrt(1f), Math.Rsqrt(1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Sin(1f), Math.Sin(1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Sqrt(1f), Math.Sqrt(1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Square(1f), Math.Square(1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Tanh(1f), Math.Tanh(1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.BitwiseNot(1f), Math.BitwiseNot(1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.LogicalNot(1f), Math.LogicalNot(1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Mod(1f, 1f), Math.Mod(1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Min(1f, 1f), Math.Min(1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Pow(1f, 1f), Math.Pow(1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.BitwiseAnd(1f, 1f), Math.BitwiseAnd(1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.BitwiseOr(1f, 1f), Math.BitwiseOr(1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.BitwiseXor(1f, 1f), Math.BitwiseXor(1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.LogicalAnd(1f, 1f), Math.LogicalAnd(1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.LogicalOr(1f, 1f), Math.LogicalOr(1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.LogicalXor(1f, 1f), Math.LogicalXor(1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.LeftShift(1f, 1f), Math.LeftShift(1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.RightShift(1f, 1f), Math.RightShift(1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.NotEqual(1f, 1f), Math.NotEqual(1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.LessThan(1f, 1f), Math.LessThan(1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.LessEqual(1f, 1f), Math.LessEqual(1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.GreaterThan(1f, 1f), Math.GreaterEqual(1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.GreaterEqual(1f, 1f), Math.GreaterThan(1f, 1f), out _));
        Assert.Equal(IR.F.Math.FloorDiv(1f, 1f), Floor(Div(1f, 1f)));
        Assert.Equal(IR.F.Math.FloorMod(1f, 1f), Sub(1f, Mul(FloorDiv(1f, 1f), 1f)));
    }

    [Fact]
    public void TestUtility()
    {
        var targetFunc = new Function(new Normal(DataTypes.Float32));
        var funcPattern = new FunctionPattern(IsOp<Normal>("normal"), IsVArgs(), null);
        Assert.True(CompilerServices.TryMatchRoot(new Call(targetFunc, 1f), IsCall(null, funcPattern, IsWildcard()), out _));
        Assert.True(CompilerServices.TryMatchRoot(new Call(targetFunc, 1f), IsCall(null, funcPattern, IsVArgs(IsWildcard())), out _));

        Assert.True(CompilerServices.TryMatchRoot(IR.F.Random.Normal(DataTypes.Float32), IsCall(null, IsOp<Normal>("normal"), IsWildcard(), IsWildcard(), IsWildcard(), IsWildcard()), out _));

        Assert.Equal(IsTensorConst(null, IsIntegral()).ToString(), IsConstIntTensor().ToString());
        Assert.Equal(IsTensorConst(null, IsIntegral()).ToString(), IsConstIntSclar().ToString());

        Assert.Equal(IsMarker(null, WellknownMarkerNames.RangeOf, IsWildcard(), IsWildcard()).Target, IsRangeOfMarker(null, IsWildcard(), IsWildcard()).Target);
        Assert.Equal(IsMarker(null, WellknownMarkerNames.RangeOf, IsWildcard(), IsWildcard()).Target, IsMarker(WellknownMarkerNames.RangeOf, IsWildcard(), IsWildcard()).Target);

        Assert.Equal(new OrPattern(IsNone(), IsNone(), string.Empty), IsAlt(string.Empty, IsNone(), IsNone()));

        Assert.Equal(new FusionPattern(null!, null!, null!, null!).Name, IsFusion<Binary, Binary, Unary>(null!, null!, null!, null!, null!, null!).Name);

        Assert.Equal(IsTupleConst(_ => true).Value, IsTupleConst().Value);

        Assert.Equal(new List<Dimension>(1), GetShape(1));

        var tuplePattern1 = new TuplePattern(new IR.Tuple(new[] { 1 }), "tuplePattern1");
        var tuplePattern2 = new TuplePattern(new IR.Tuple(new[] { 2 }), "tuplePattern2");
        Assert.NotEqual(tuplePattern1, tuplePattern2);

        var tupleConstPattern1 = new TupleConstPattern(new TupleConst(new TupleValue(new[] { Value.FromConst(1F) })), null);
        var tupleConstPattern2 = new TupleConstPattern(new TupleConst(new TupleValue(new[] { Value.FromConst(2F) })), null);
        Assert.NotEqual(tupleConstPattern1, tupleConstPattern2);

        var markerPattern1 = new MarkerPattern(new Marker(null!, 1, 1), null);
        var markerPattern2 = new MarkerPattern(new Marker(null!, 2, 1), null);
        Assert.NotEqual(markerPattern1, markerPattern2);

        var expect = IsAlt(
            IsBinary("testName", "call", binary => true, 1f, 1f),
            IsBinary("testName", "call", binary => true, 1f, 1f)).Name;
        Assert.Equal(expect, IsSwappableBinary("testName", "call", binary => true, 1f, 1f).Name);
    }

    [Fact]
    public void TestCallPattern()
    {
        var func = IR.F.Random.Normal(DataTypes.Float32);
        Assert.Throws<MissingMethodException>(() => new CallPattern(func, null));
    }
}
