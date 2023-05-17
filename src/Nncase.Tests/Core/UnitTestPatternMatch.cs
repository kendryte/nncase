// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Extensions.Hosting;
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
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Abs(1f), IsUnary(UnaryOp.Abs, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Ceil(1f), IsUnary(UnaryOp.Ceil, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Cos(1f), IsUnary(UnaryOp.Cos, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Exp(1f), IsUnary(UnaryOp.Exp, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Floor(1f), IsUnary(UnaryOp.Floor, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Log(1f), IsUnary(UnaryOp.Log, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Neg(1f), IsUnary(UnaryOp.Neg, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Round(1f), IsUnary(UnaryOp.Round, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Rsqrt(1f), IsUnary(UnaryOp.Rsqrt, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Sin(1f), IsUnary(UnaryOp.Sin, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Sqrt(1f), IsUnary(UnaryOp.Sqrt, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Square(1f), IsUnary(UnaryOp.Square, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Tanh(1f), IsUnary(UnaryOp.Tanh, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.BitwiseNot(1f), IsUnary(UnaryOp.BitwiseNot, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.LogicalNot(1f), IsUnary(UnaryOp.LogicalNot, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Mod(1f, 1f), IsBinary(BinaryOp.Mod, 1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Min(1f, 1f), IsBinary(BinaryOp.Min, 1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.Pow(1f, 1f), IsBinary(BinaryOp.Pow, 1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.BitwiseAnd(1f, 1f), IsBinary(BinaryOp.BitwiseAnd, 1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.BitwiseOr(1f, 1f), IsBinary(BinaryOp.BitwiseOr, 1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.BitwiseXor(1f, 1f), IsBinary(BinaryOp.BitwiseXor, 1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.LogicalAnd(1f, 1f), IsBinary(BinaryOp.LogicalAnd, 1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.LogicalOr(1f, 1f), IsBinary(BinaryOp.LogicalOr, 1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.LogicalXor(1f, 1f), IsBinary(BinaryOp.LogicalXor, 1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.LeftShift(1f, 1f), IsBinary(BinaryOp.LeftShift, 1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.RightShift(1f, 1f), IsBinary(BinaryOp.RightShift, 1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.NotEqual(1f, 1f), IsCompare(CompareOp.NotEqual, 1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.LessThan(1f, 1f), IsCompare(CompareOp.LowerThan, 1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.LessEqual(1f, 1f), IsCompare(CompareOp.LowerOrEqual, 1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.GreaterThan(1f, 1f), IsCompare(CompareOp.GreaterThan, 1f, 1f), out _));
        Assert.True(CompilerServices.TryMatchRoot(IR.F.Math.GreaterEqual(1f, 1f), IsCompare(CompareOp.GreaterOrEqual, 1f, 1f), out _));
        Assert.Equal(IR.F.Math.FloorDiv(1f, 1f), Floor(Div(1f, 1f)));
        Assert.Equal(IR.F.Math.FloorMod(1f, 1f), Sub(1f, Mul(FloorDiv(1f, 1f), 1f)));
    }

    [Fact]
    public void TestUtility()
    {
        IsSwappableBinary("testName", "call", binary => true, 1f, 1f);

        var targetFunc = new Function(new Normal(DataTypes.Float32));
        var funcPattern = new FunctionPattern(IsOp<Normal>("normal"), IsVArgs(), null);
        Assert.True(CompilerServices.TryMatchRoot(new Call(targetFunc, 1f), IsCall(null, funcPattern, IsWildcard()), out _));
        Assert.True(CompilerServices.TryMatchRoot(new Call(targetFunc, 1f), IsCall(null, funcPattern, IsVArgs(IsWildcard())), out _));

        Assert.True(CompilerServices.TryMatchRoot(IR.F.Random.Normal(DataTypes.Float32), IsCall(null, IsOp<Normal>("normal"), IsWildcard(), IsWildcard(), IsWildcard(), IsWildcard()), out _));

        Assert.True(CompilerServices.TryMatchRoot(new Call(new TensorConst(null)), IsConstIntTensor(null), out _));
        Assert.NotNull(IsConstIntTensor(null));
        Assert.NotNull(IsConstIntTensor());
        Assert.NotNull(IsConstIntSclar());

        Assert.NotNull(IsRangeOfMarker(null, IsWildcard(), IsWildcard()));
        Assert.NotNull(IsMarker(WellknownMarkerNames.RangeOf, IsWildcard(), IsWildcard()));

        Assert.NotNull(IsAlt(string.Empty, IsNone(), IsNone()));

        Assert.NotNull(IsFusion<Binary, Binary, Unary>(null!, null!, null!, null!, null!, null!));

        Assert.NotNull(IsTupleConst());
        Assert.NotNull(IsTupleConst(_ => true));

        Assert.Equal(new List<Dimension>(1), GetShape(1));

        var tuplePattern = new TuplePattern(new IR.Tuple(new[] { 1 }), null);
        Assert.NotNull(tuplePattern);

        var tupleConstPattern = new TupleConstPattern(new TupleConst(new TupleValue(new[] { Value.FromConst(1F) })), null);
        Assert.NotNull(tupleConstPattern);

        var markerPattern = new MarkerPattern(new Marker(null!, 1, 1), null);
        Assert.NotNull(markerPattern);
    }

    [Fact]
    public void TestCallPattern()
    {
        var func = IR.F.Random.Normal(DataTypes.Float32);
        Assert.Throws<MissingMethodException>(() => new CallPattern(func, null));
    }
}
