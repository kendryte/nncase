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
    public void TestDistributedTypeMemoryAccessCost1()
    {
        var t = new DistributedType(new TensorType(DataTypes.Float32, new[] { 64, 3 }), new SBP[] { SBP.S(0), SBP.S(0) }, new Placement(new[] { 16, 4 }, "bt"));
        var cost = CostUtility.GetMemoryAccess(t);
        System.Console.WriteLine(cost);
        Assert.True(cost == 12);
    }
}
