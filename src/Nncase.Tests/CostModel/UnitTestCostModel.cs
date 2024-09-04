// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase;
using Nncase.CostModel;
using Nncase.IR;
using Xunit;

namespace Nncase.Tests.CostModelTest;

public class UnitTestCostModel
{
    [Fact]
    public void TestDistributedTypeMemoryAccessCost()
    {
        var t1 = new DistributedType(new TensorType(DataTypes.Float32, new[] { 2, 4096, 320 }), new SBP[] { SBP.S(2), SBP.S(1) }, new Placement(new[] { 8, 4 }, "bt"));
        var cost1 = CostModel.CostUtility.GetMemoryAccess(t1);
        var t2 = new DistributedType(new TensorType(DataTypes.Float32, new[] { 2, 4096, 320 }), new SBP[] { SBP.B, SBP.B }, new Placement(new[] { 8, 4 }, "bt"));
        var cost2 = CostModel.CostUtility.GetMemoryAccess(t2);
        System.Console.WriteLine(cost1);
        System.Console.WriteLine(cost2);
        Assert.True(cost1 < cost2);
    }

    [Fact]
    public void TestDistributedTypeMemoryAccessCost1()
    {
        var t = new DistributedType(new TensorType(DataTypes.Float32, new[] { 64, 3 }), new SBP[] { SBP.S(0), SBP.S(0) }, new Placement(new[] { 16, 4 }, "bt"));
        var cost = CostUtility.GetMemoryAccess(t);
        System.Console.WriteLine(cost);
        Assert.True(cost == 12);
    }
}
