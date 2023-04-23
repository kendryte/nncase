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
using Nncase.PatternMatch.F;
using Tensorflow;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using Function = Nncase.IR.Function;
using Math = Nncase.PatternMatch.F.Math;

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
        var input = IR.F.Random.Normal(new[] { 1, 1, 1, 1 });
        Math.Abs(1f);
        Math.Ceil(1f);
        Math.Cos(1f);
        Math.Exp(1f);
        Math.Floor(1f);
        Math.Log(1f);
        Math.Neg(1f);
        Math.Round(1f);
        Math.Rsqrt(1f);
        Math.Sin(1f);
        Math.Sqrt(1f);
        Math.Square(1f);
        Math.Tanh(1f);
        Math.BitwiseNot(1f);
        Math.LogicalNot(1f);
        Math.Mod(1f, 1f);
        Math.Min(1f, 1f);
        Math.Max(1f, 1f);
        Math.Pow(1f, 1f);
        Math.BitwiseAnd(1f, 1f);
        Math.BitwiseOr(1f, 1f);
        Math.BitwiseXor(1f, 1f);
        Math.LogicalAnd(1f, 1f);
        Math.LogicalOr(1f, 1f);
        Math.LogicalXor(1f, 1f);
        Math.LeftShift(1f, 1f);
        Math.RightShift(1f, 1f);
        Math.FloorDiv(1f, 1f);
        Math.FloorMod(1f, 1f);
        Math.NotEqual(1f, 1f);
        Math.LessThan(1f, 1f);
        Math.LessEqual(1f, 1f);
        Math.GreaterEqual(1f, 1f);
        Math.GreaterThan(1f, 1f);
    }
}
