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
using Nncase.Passes;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;
using Nncase.TIR;
using Tensorflow;
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
        // Assert.False(Math.Abs(1f).GetHashCode().Equals(IsUnary(UnaryOp.Abs, 1f).GetHashCode()));
        // Assert.False(Math.Ceil(1f).GetHashCode().Equals(IsUnary(UnaryOp.Ceil, 1f).GetHashCode()));
        // Assert.False(Math.Cos(1f).GetHashCode().Equals(IsUnary(UnaryOp.Cos, 1f).GetHashCode()));
        // Assert.False(Math.Exp(1f).GetHashCode().Equals(IsUnary(UnaryOp.Exp, 1f).GetHashCode()));
        // Assert.False(Math.Floor(1f).GetHashCode().Equals(IsUnary(UnaryOp.Floor, 1f).GetHashCode()));
        // Assert.False(Math.Log(1f).GetHashCode().Equals(IsUnary(UnaryOp.Log, 1f).GetHashCode()));
        // Assert.False(Math.Neg(1f).GetHashCode().Equals(IsUnary(UnaryOp.Neg, 1f).GetHashCode()));
        // Assert.False(Math.Round(1f).GetHashCode().Equals(IsUnary(UnaryOp.Round, 1f).GetHashCode()));
        // Assert.False(Math.Rsqrt(1f).GetHashCode().Equals(IsUnary(UnaryOp.Rsqrt, 1f).GetHashCode()));
        // Assert.False(Math.Sin(1f).GetHashCode().Equals(IsUnary(UnaryOp.Sin, 1f).GetHashCode()));
        // Assert.False(Math.Sqrt(1f).GetHashCode().Equals(IsUnary(UnaryOp.Sqrt, 1f).GetHashCode()));
        // Assert.False(Math.Square(1f).GetHashCode().Equals(IsUnary(UnaryOp.Square, 1f).GetHashCode()));
        // Assert.False(Math.Tanh(1f).GetHashCode().Equals(IsUnary(UnaryOp.Tanh, 1f).GetHashCode()));
        // Assert.False(Math.BitwiseNot(1f).GetHashCode().Equals(IsUnary(UnaryOp.BitwiseNot, 1f).GetHashCode()));
        // Assert.False(Math.LogicalNot(1f).GetHashCode().Equals(IsUnary(UnaryOp.LogicalNot, 1f).GetHashCode()));
        // Assert.False(Math.Mod(1f, 1f).GetHashCode().Equals(IsBinary(BinaryOp.Mod, 1f, 1f).GetHashCode()));
        // Assert.False(Math.Min(1f, 1f).GetHashCode().Equals(IsBinary(BinaryOp.Min, 1f, 1f).GetHashCode()));
        // Assert.False(Math.Pow(1f, 1f).GetHashCode().Equals(IsBinary(BinaryOp.Pow, 1f, 1f).GetHashCode()));
        // Assert.False(Math.BitwiseAnd(1f, 1f).GetHashCode().Equals(IsBinary(BinaryOp.BitwiseAnd, 1f, 1f).GetHashCode()));
        // Assert.False(Math.BitwiseOr(1f, 1f).GetHashCode().Equals(IsBinary(BinaryOp.BitwiseOr, 1f, 1f).GetHashCode()));
        // Assert.False(Math.BitwiseXor(1f, 1f).GetHashCode().Equals(IsBinary(BinaryOp.BitwiseXor, 1f, 1f).GetHashCode()));
        // Assert.False(Math.LogicalAnd(1f, 1f).GetHashCode().Equals(IsBinary(BinaryOp.LogicalAnd, 1f, 1f).GetHashCode()));
        // Assert.False(Math.LogicalOr(1f, 1f).GetHashCode().Equals(IsBinary(BinaryOp.LogicalOr, 1f, 1f).GetHashCode()));
        // Assert.False(Math.LogicalXor(1f, 1f).GetHashCode().Equals(IsBinary(BinaryOp.LogicalXor, 1f, 1f).GetHashCode()));
        // Assert.False(Math.LeftShift(1f, 1f).GetHashCode().Equals(IsBinary(BinaryOp.LeftShift, 1f, 1f).GetHashCode()));
        // Assert.False(Math.RightShift(1f, 1f).GetHashCode().Equals(IsBinary(BinaryOp.RightShift, 1f, 1f).GetHashCode()));
        // Assert.False(Math.FloorDiv(1f, 1f).GetHashCode().Equals(Floor(1f / 1f).GetHashCode()));
        // Assert.False(Math.FloorMod(1f, 1f).GetHashCode().Equals(Sub(1f, FloorDiv(1f, 1f) * 1f).GetHashCode()));
        // Assert.False(Math.NotEqual(1f, 1f).GetHashCode().Equals(IsCompare(CompareOp.NotEqual, 1f, 1f).GetHashCode()));
        // Assert.False(Math.LessThan(1f, 1f).GetHashCode().Equals(IsCompare(CompareOp.LowerThan, 1f, 1f).GetHashCode()));
        // Assert.False(Math.LessEqual(1f, 1f).GetHashCode().Equals(IsCompare(CompareOp.LowerOrEqual, 1f, 1f).GetHashCode()));
        // Assert.False(Math.GreaterEqual(1f, 1f).GetHashCode().Equals(IsCompare(CompareOp.GreaterThan, 1f, 1f).GetHashCode()));
        // Assert.False(Math.GreaterThan(1f, 1f).GetHashCode().Equals(IsCompare(CompareOp.GreaterOrEqual, 1f, 1f).GetHashCode()));
    }

    [Fact]
    public void TestUtility()
    {
        IsSwappableBinary("testName", "call", binary => true, 1f, 1f);

        var wc1 = IsWildcard();
        var wc2 = IsWildcard();
        Assert.NotNull(IsCall(null, new FunctionPattern(wc1 + wc2, IsVArgs(wc1, wc2), null)));
        Assert.NotNull(IsCall(null, new FunctionPattern(wc1 + wc2, IsVArgs(wc1, wc2), null), IsVArgs(wc1, wc2)));
        Assert.NotNull(IsCall("call", IsOp<ActivationOp>("activation", op => true)));

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
