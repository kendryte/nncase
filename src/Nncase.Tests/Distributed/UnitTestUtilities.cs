// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.DistributedTest;

public sealed class UnitTestUtilities
{
    [Fact]
    public void TestEffiecicy()
    {
        var burst = 256;
        var type1 = new DistributedType(new TensorType(DataTypes.Float32, new[] { 1, 64, 384, 8192 }), new[] { SBP.S(1), SBP.S(2) }, new(Placement.DeviceKind.CPU, new[] { 8, 4 }, "bt"));
        var eff1 = DistributedUtility.GetDividedTensorEfficiency(type1, burst);

        var type2 = new DistributedType(new TensorType(DataTypes.Float32, new[] { 1, 64, 384, 8192 }), new[] { SBP.S(1), SBP.S(3) }, new(Placement.DeviceKind.CPU, new[] { 8, 4 }, "bt"));
        var eff2 = DistributedUtility.GetDividedTensorEfficiency(type2, burst);

        var type3 = new DistributedType(new TensorType(DataTypes.Float32, new[] { 1, 64, 384, 8192 }), new[] { SBP.S(3), SBP.S(3) }, new(Placement.DeviceKind.CPU, new[] { 8, 4 }, "bt"));
        var eff3 = DistributedUtility.GetDividedTensorEfficiency(type3, burst);
        Assert.True(eff1 > eff2);
        Assert.True(eff2 > eff3);
    }

    [Fact]
    public void TestGetPartialCandidateNDSBPs()
    {
        var placement = new Placement(Placement.DeviceKind.CPU, new[] { 8, 4 }, "bt");
        var type = new DistributedType(new(DataTypes.Float32, new[] { 1, 384, 8192 }), new SBP[] { SBP.P, SBP.S(2) }, placement);
        var candidateSbps = Utilities.DistributedUtility.GetPartialCandidateNDSBPs(type);

        Assert.Equal(0, candidateSbps.Count(ndsbp => ndsbp == new IRArray<SBP>(new SBP[] { SBP.S(2), SBP.S(2) })));
    }
}
