// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes.Distributed;
using Nncase.Tests.TestFixture;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.DistributedTest;

public sealed class UnitTestDistributedUtilities
{
    public static TheoryData<DistributedType, DistributedType, bool> FoldBoxingReshapeData { get; } = new()
    {
        {
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 1024 }), new SBP[] { SBP.S(2) }, new(new[] { 8 }, "t")),
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 64, 16 }), new SBP[] { SBP.S(2) }, new(new[] { 8 }, "t")),
            true
        },
        {
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 1024 }), new SBP[] { SBP.S(1) }, new(new[] { 8 }, "t")),
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 64, 16 }), new SBP[] { SBP.S(1) }, new(new[] { 8 }, "t")),
            true
        },

        // resplit on second splited-by-reshape axis.
        {
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 1024 }), new SBP[] { SBP.S(2) }, new(new[] { 8 }, "t")),
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 64, 16 }), new SBP[] { SBP.S(3) }, new(new[] { 8 }, "t")),
            false
        },

        // resplit on first merged-by-reshape axis.
        {
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 64, 16 }), new SBP[] { SBP.S(2) }, new(new[] { 8 }, "t")),
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 1024 }), new SBP[] { SBP.S(2) }, new(new[] { 8 }, "t")),
            true
        },

        // resplit on second merged-by-reshape axis.
        {
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 64, 16 }), new SBP[] { SBP.S(3) }, new(new[] { 8 }, "t")),
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 1024 }), new SBP[] { SBP.S(2) }, new(new[] { 8 }, "t")),
            false
        },

        // resplit on first merged-by-reshape axis.
        {
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 64, 16 }), new SBP[] { SBP.S(1), SBP.S(2) }, new(new[] { 8 }, "t")),
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 1024 }), new SBP[] { SBP.S(1), SBP.S(2) }, new(new[] { 8 }, "t")),
            true
        },

        // resplit on second merged-by-reshape axis.
        {
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 64, 16 }), new SBP[] { SBP.S(1), SBP.S(3) }, new(new[] { 8 }, "t")),
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 1024 }), new SBP[] { SBP.S(1), SBP.S(2) }, new(new[] { 8 }, "t")),
            false
        },

        // resplit on second merged-by-reshape axis.
        {
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 64, 16 }), new SBP[] { SBP.S(1), SBP.S(3) }, new(new[] { 8 }, "t")),
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 1024 }), new SBP[] { SBP.S(1), SBP.S(2) }, new(new[] { 8 }, "t")),
            false
        },

        // resplit on different mesh axis.
        {
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 64, 16 }), new SBP[] { SBP.S(1), SBP.S(2) }, new(new[] { 8 }, "t")),
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 1024 }), new SBP[] { SBP.S(2), SBP.S(1) }, new(new[] { 8 }, "t")),
            false
        },
    };

    [Fact]
    public void TestGenerateReduceGroups()
    {
        Assert.Single(Utilities.LinqUtility.Combination(1));
        Assert.Equal(3, Utilities.LinqUtility.Combination(2).Count());
    }

    [Theory]
    [MemberData(nameof(FoldBoxingReshapeData))]
    public void TestFoldBoxingReshape(DistributedType inType, DistributedType outType, bool excepted)
    {
        Assert.Equal(excepted, DistributedUtility.IsNoReshardReshape(inType, outType));
    }
}
