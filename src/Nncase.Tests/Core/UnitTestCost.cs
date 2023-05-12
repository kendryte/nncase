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
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Abs));
        Assert.Equal((UInt128)8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Acos));
        Assert.Equal((UInt128)8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Acosh));
        Assert.Equal((UInt128)8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Asin));
        Assert.Equal((UInt128)8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Asinh));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Ceil));
        Assert.Equal((UInt128)8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Cos));
        Assert.Equal((UInt128)8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Cosh));
        Assert.Equal((UInt128)8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Exp));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Floor));
        Assert.Equal((UInt128)8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Log));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Neg));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Round));
        Assert.Equal((UInt128)4, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Rsqrt));
        Assert.Equal((UInt128)8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Sin));
        Assert.Equal((UInt128)8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Sinh));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Sign));
        Assert.Equal((UInt128)8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Sqrt));
        Assert.Equal((UInt128)2, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Square));
        Assert.Equal((UInt128)8, CostUtility.GetCPUCyclesOfUnary(UnaryOp.Tanh));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfUnary(UnaryOp.BitwiseNot));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfUnary(UnaryOp.LogicalNot));
    }

    [Fact]
    public void TestGetCPUCyclesOfBinary()
    {
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Add));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Sub));
        Assert.Equal((UInt128)2, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Mul));
        Assert.Equal((UInt128)8, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Div));
        Assert.Equal((UInt128)8, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Mod));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Min));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Max));
        Assert.Equal((UInt128)8, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Pow));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.BitwiseAnd));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.BitwiseOr));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.BitwiseXor));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.LogicalAnd));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.LogicalOr));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.LogicalXor));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.LeftShift));
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfBinary(BinaryOp.RightShift));
    }

    [Fact]
    public void TestGetCPUCyclesOfMax()
    {
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfMax());
    }

    [Fact]
    public void TestGetCPUCyclesOfCompare()
    {
        Assert.Equal((UInt128)1, CostUtility.GetCPUCyclesOfCompare());
    }

    [Fact]
    public void TestGetReshapeCost()
    {
        var cost = new Cost
        {
            [CostFactorNames.CPUCycles] = 1,
        };
        Assert.Equal(cost, CostUtility.GetReshapeCost());
    }

    [Fact]
    public void TestGetBroadcastCost()
    {
        var cost = new Cost()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(new TensorType(DataTypes.Int32, new[] { 1, 2, 4, 8 })),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(new TensorType(DataTypes.Int32, new[] { 1, 2, 4, 8 })),
            [CostFactorNames.CPUCycles] = 1,
        };
        Assert.Equal(cost, CostUtility.GetBroadcastCost(new TensorType(DataTypes.Int32, new[] { 1, 2, 4, 8 }), new TensorType(DataTypes.Int32, new[] { 1, 2, 4, 8 })));
    }

    [Fact]
    public void TestGetCPUCycles()
    {
        Assert.Equal((UInt128)1, CostUtility.GetCPUCycles(DataTypes.Int32));
        Assert.Equal((UInt128)0, CostUtility.GetCPUCycles(new TupleType(new List<IRType>())));
    }

    [Fact]
    public void TestGetFakeMemoryAccess()
    {
        Assert.Equal((UInt128)0, CostUtility.GetFakeMemoryAccess(DataTypes.Float32, 0));
        Assert.Equal((UInt128)0, CostUtility.GetFakeMemoryAccess(new TupleType(new List<IRType>()), 0));
    }

    [Fact]
    public void TestGetMemoryAccess()
    {
        Assert.Equal((UInt128)8, CostUtility.GetMemoryAccess(DataTypes.Float32, DataTypes.Float32));
    }

    [Fact]
    public void TestDoubleCostSumErrorCase()
    {
        // note when use double costs, somtime the value sum will got error.
        var c = new Cost()
        {
            [CostFactorNames.MemoryLoad] = (UInt128)1651643172136040,
            [CostFactorNames.MemoryStore] = (UInt128)953885239498522,
            [CostFactorNames.CPUCycles] = (UInt128)13437180688291688,
        };

        var c2 = new Cost()
        {
            [CostFactorNames.CPUCycles] = UInt128.One,
        };

        Assert.Equal((UInt128)16042709099926251, (c2 + c).Score);
    }
}
