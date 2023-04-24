// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using Microsoft.VisualBasic.CompilerServices;
using Nncase;
using Nncase.CostModel;
using Nncase.IR;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestCost
{
    [Fact]
    public void TestGetCPUCyclesOfUnary()
    {
        Assert.Equal(1, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Abs));
        Assert.Equal(8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Acos));
        Assert.Equal(8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Acosh));
        Assert.Equal(8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Asin));
        Assert.Equal(8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Asinh));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Ceil));
        Assert.Equal(8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Cos));
        Assert.Equal(8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Cosh));
        Assert.Equal(8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Exp));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Floor));
        Assert.Equal(8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Log));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Neg));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Round));
        Assert.Equal(4, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Rsqrt));
        Assert.Equal(8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Sin));
        Assert.Equal(8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Sinh));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Sign));
        Assert.Equal(8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Sqrt));
        Assert.Equal(2, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Square));
        Assert.Equal(8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Tanh));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfUnary(UnaryOp.BitwiseNot));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfUnary(UnaryOp.LogicalNot));
    }

    [Fact]
    public void TestGetCPUCyclesOfBinary()
    {
        Assert.Equal(1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Add));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Sub));
        Assert.Equal(2, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Mul));
        Assert.Equal(8, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Div));
        Assert.Equal(8, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Mod));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Min));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Max));
        Assert.Equal(8, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Pow));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.BitwiseAnd));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.BitwiseOr));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.BitwiseXor));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.LogicalAnd));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.LogicalOr));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.LogicalXor));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.LeftShift));
        Assert.Equal(1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.RightShift));
    }
}
