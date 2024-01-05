// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using Nncase;
using Nncase.IR;
using Nncase.IR.Random;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestMath
{
    [Fact]
    public void TestMinMax()
    {
        var input = IR.F.Random.Normal(new[] { 1 });
        var min = IR.F.Random.Normal(new[] { 1 });
        var max = IR.F.Random.Normal(new[] { 1 });
        var expr = IR.F.Math.MinMax(input, min, max);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Min(IR.F.Math.Max(input, min), max);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr.Evaluate(), expect.Evaluate());
    }

    [Fact]
    public void TestCos()
    {
        var input = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.Cos(input);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Unary(UnaryOp.Cos, input);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr.Evaluate(), expect.Evaluate());
    }

    [Fact]
    public void TestFloor()
    {
        var input = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.Floor(input);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Unary(UnaryOp.Floor, input);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr.Evaluate(), expect.Evaluate());
    }

    [Fact]
    public void TestLog()
    {
        var input = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.Log(input);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Unary(UnaryOp.Log, input);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr.Evaluate(), expect.Evaluate());
    }

    [Fact]
    public void TestRound()
    {
        var input = IR.F.Random.Normal(new[] { 1, 3 });
        var expr = IR.F.Math.Round(input);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Unary(UnaryOp.Round, input);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr.Evaluate(), expect.Evaluate());
    }

    [Fact]
    public void TestRsqrt()
    {
        var input = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.Rsqrt(input);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Unary(UnaryOp.Rsqrt, input);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr.Evaluate(), expect.Evaluate());
    }

    [Fact]
    public void TestSin()
    {
        var input = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.Sin(input);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Unary(UnaryOp.Sin, input);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr.Evaluate(), expect.Evaluate());
    }

    [Fact]
    public void TestTanh()
    {
        var input = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.Tanh(input);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Unary(UnaryOp.Tanh, input);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr.Evaluate(), expect.Evaluate());
    }

    [Fact]
    public void TestBitwiseNot()
    {
        var input = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.BitwiseNot(input);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Unary(UnaryOp.BitwiseNot, input);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr, expect);
    }

    [Fact]
    public void TestLogicalNot()
    {
        var input = IR.F.Random.Normal(new[] { 1, 3 });
        var expr = IR.F.Math.LogicalNot(input);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Unary(UnaryOp.LogicalNot, input);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr, expect);
    }

    [Fact]
    public void TestBitwiseAnd()
    {
        var lhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var rhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.BitwiseAnd(lhs, rhs);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Binary(BinaryOp.BitwiseAnd, lhs, rhs);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr, expect);
    }

    [Fact]
    public void TestBitwiseOr()
    {
        var lhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var rhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.BitwiseOr(lhs, rhs);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Binary(BinaryOp.BitwiseOr, lhs, rhs);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr, expect);
    }

    [Fact]
    public void TestBitwiseXor()
    {
        var lhs = IR.F.Random.Normal(DataTypes.Int8, new[] { 1, 3, 16, 16 });
        var rhs = IR.F.Random.Normal(DataTypes.Int8, new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.BitwiseXor(lhs, rhs);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Binary(BinaryOp.BitwiseXor, lhs, rhs);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr, expect);
    }

    [Fact]
    public void TestLogicalAnd()
    {
        var lhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var rhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.LogicalAnd(lhs, rhs);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Binary(BinaryOp.LogicalAnd, lhs, rhs);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr, expect);
    }

    [Fact]
    public void TestLogicalOr()
    {
        var lhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var rhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.LogicalOr(lhs, rhs);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Binary(BinaryOp.LogicalOr, lhs, rhs);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr, expect);
    }

    [Fact]
    public void TestLogicalXor()
    {
        var lhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var rhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.LogicalXor(lhs, rhs);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Binary(BinaryOp.LogicalXor, lhs, rhs);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr, expect);
    }

    [Fact]
    public void TestRightShift()
    {
        var lhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var rhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.RightShift(lhs, rhs);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Binary(BinaryOp.RightShift, lhs, rhs);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr, expect);
    }

    [Fact]
    public void TestFloorDiv()
    {
        var lhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var rhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.FloorDiv(lhs, rhs);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Binary(BinaryOp.FloorDiv, lhs, rhs);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr, expect);
    }

    [Fact]
    public void TestFloorMod()
    {
        var lhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var rhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.FloorMod(lhs, rhs);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Sub(lhs, IR.F.Math.FloorDiv(lhs, rhs) * rhs);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr, expect);
    }

    [Fact]
    public void TestMod()
    {
        var lhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var rhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.Binary(BinaryOp.Mod, lhs, rhs);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Mod(lhs, rhs);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expect.Evaluate(), expr.Evaluate());
    }

    [Fact]
    public void TestNotEqual()
    {
        var lhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var rhs = IR.F.Random.Normal(new[] { 1, 3, 16, 16 });
        var expr = IR.F.Math.NotEqual(lhs, rhs);
        CompilerServices.InferenceType(expr);
        var expect = IR.F.Math.Compare(CompareOp.NotEqual, lhs, rhs);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr, expect);
    }
}
