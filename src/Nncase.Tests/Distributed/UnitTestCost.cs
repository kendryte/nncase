// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Xunit;

namespace Nncase.Tests.DistributedTest;

public sealed class UnitTestCost
{
    [Fact]
    public void TestPartialToBroadCastCost()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 1, 64, 384, 128 });
        var placement = new Placement(new[] { 8, 4 }, "bt");

        List<CostModel.Cost> costs = new();
        foreach (var ndsbp in new IRArray<SBP>[] {
            new SBP[] { SBP.S(2), SBP.P },
            new SBP[] { SBP.P, SBP.S(2) },
            new SBP[] { SBP.P, SBP.P },
            new SBP[] { SBP.P, SBP.S(1) },
            new SBP[] { SBP.P, SBP.S(3) },
            new SBP[] { SBP.S(1), SBP.P },
            new SBP[] { SBP.S(3), SBP.P },
         })
        {
            var type = new DistributedType(tensorType, ndsbp, placement);
            var input0 = new Var(type);
            var candidates = Utilities.DistributedUtility.GetPartialCandidateNDSBPs(type).Select(ndsbp => IR.F.XPU.Boxing(input0, type with { NdSBP = ndsbp }));
            foreach (var item in candidates)
            {
                var cost = CompilerServices.EvaluateCost(item);
                costs.Add(cost);
            }
        }
    }
}
