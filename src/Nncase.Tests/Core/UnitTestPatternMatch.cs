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
        Assert.NotNull(Math.Abs(1f));
        Assert.NotNull(Math.Ceil(1f));
        Assert.NotNull(Math.Cos(1f));
        Assert.NotNull(Math.Exp(1f));
        Assert.NotNull(Math.Floor(1f));
        Assert.NotNull(Math.Log(1f));
        Assert.NotNull(Math.Neg(1f));
        Assert.NotNull(Math.Round(1f));
        Assert.NotNull(Math.Rsqrt(1f));
        Assert.NotNull(Math.Sin(1f));
        Assert.NotNull(Math.Sqrt(1f));
        Assert.NotNull(Math.Square(1f));
        Assert.NotNull(Math.Tanh(1f));
        Assert.NotNull(Math.BitwiseNot(1f));
        Assert.NotNull(Math.LogicalNot(1f));
        Assert.NotNull(Math.Mod(1f, 1f));
        Assert.NotNull(Math.Min(1f, 1f));
        Assert.NotNull(Math.Max(1f, 1f));
        Assert.NotNull(Math.Pow(1f, 1f));
        Assert.NotNull(Math.BitwiseAnd(1f, 1f));
        Assert.NotNull(Math.BitwiseOr(1f, 1f));
        Assert.NotNull(Math.BitwiseXor(1f, 1f));
        Assert.NotNull(Math.LogicalAnd(1f, 1f));
        Assert.NotNull(Math.LogicalOr(1f, 1f));
        Assert.NotNull(Math.LogicalXor(1f, 1f));
        Assert.NotNull(Math.LeftShift(1f, 1f));
        Assert.NotNull(Math.RightShift(1f, 1f));
        Assert.NotNull(Math.FloorDiv(1f, 1f));
        Assert.NotNull(Math.FloorMod(1f, 1f));
        Assert.NotNull(Math.NotEqual(1f, 1f));
        Assert.NotNull(Math.LessThan(1f, 1f));
        Assert.NotNull(Math.LessEqual(1f, 1f));
        Assert.NotNull(Math.GreaterEqual(1f, 1f));
        Assert.NotNull(Math.GreaterThan(1f, 1f));
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
