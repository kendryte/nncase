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

public sealed class UnitTestDistributedTypeInfer
{
    public static TheoryData<DistributedType, int[], IRType> ReshapeTypeInferData { get; } = new()
    {
        {
            // split on not related axis.
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 1024 }), new SBP[] { SBP.S(1) }, new(new[] { 8 }, "t")),
            new[] { 1, 48, 64, 16 },
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 64, 16 }), new SBP[] { SBP.S(1) }, new(new[] { 8 }, "t"))
        },
        {
            // split on splited-by-reshape axis.
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 1024 }), new SBP[] { SBP.S(2) }, new(new[] { 8 }, "t")),
            new[] { 1, 48, 64, 16 },
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 64, 16 }), new SBP[] { SBP.S(2) }, new(new[] { 8 }, "t"))
        },
        {
            // split on sequeezed axis.
            new DistributedType(new(DataTypes.Float32, new[] { 1, 1, 48, 1024 }), new SBP[] { SBP.S(2) }, new(new[] { 8 }, "t")),
            new[] { 1, 48, 64, 16 },
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 64, 16 }), new SBP[] { SBP.S(1) }, new(new[] { 8 }, "t"))
        },
        {
            // split on right unsequeeze axis.
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 1024 }), new SBP[] { SBP.S(1) }, new(new[] { 8 }, "t")),
            new[] { 1, 48, 1, 64, 16 },
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 1, 64, 16 }), new SBP[] { SBP.S(1) }, new(new[] { 8 }, "t"))
        },
        {
            // split on left unsequeeze axis.
            new DistributedType(new(DataTypes.Float32, new[] { 1, 384, 128 }), new SBP[] { SBP.S(1) }, new(new[] { 8 }, "t")),
            new[] { 1, 1, 384, 128 },
            new DistributedType(new(DataTypes.Float32, new[] { 1, 1, 384, 128 }), new SBP[] { SBP.S(2) }, new(new[] { 8 }, "t"))
        },
        {
            // split on merged-by-reshape axis.
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 64, 16 }), new SBP[] { SBP.S(2) }, new(new[] { 8 }, "t")),
            new[] { 1, 48, 1024 },
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 1024 }), new SBP[] { SBP.S(2) }, new(new[] { 8 }, "t"))
        },
        {
            // split on merged-by-reshape axis, but not support.
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 64, 16 }), new SBP[] { SBP.S(3) }, new(new[] { 8 }, "t")),
            new[] { 1, 48, 1024 },
            new InvalidType("not support")
        },
        {
            // mesh dim 0 split on first merged-by-reshape axis.
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 64, 16 }), new SBP[] { SBP.S(1), SBP.S(2) }, new(new[] { 8 }, "t")),
            new[] { 1, 48, 1024 },
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 1024 }), new SBP[] { SBP.S(1), SBP.S(2) }, new(new[] { 8 }, "t"))
        },
        {
            // mesh dim 1 split on first merged-by-reshape axis.
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 64, 16 }), new SBP[] { SBP.S(2), SBP.S(1),  }, new(new[] { 8 }, "t")),
            new[] { 1, 48, 1024 },
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 1024 }), new SBP[] { SBP.S(2), SBP.S(1), }, new(new[] { 8 }, "t"))
        },
        {
            // split on second merged-by-reshape axis.
            new DistributedType(new(DataTypes.Float32, new[] { 1, 48, 64, 16 }), new SBP[] { SBP.S(1), SBP.S(3) }, new(new[] { 8 }, "t")),
            new[] { 1, 48, 1024 },
            new InvalidType("not support")
        },
        {
            // unmapable reshape
            new DistributedType(new(DataTypes.Float32, new[] { 2, 30 }), new SBP[] { SBP.S(0) }, new(new[] { 6 }, "t")),
            new[] { 3, 20 },
            new InvalidType("unmapable")
        },
    };

    [Fact]
    public void TestGenerateReduceGroups()
    {
        Assert.Single(LinqUtility.Combination(1));
        Assert.Equal(3, LinqUtility.Combination(2).Count());
    }

    [Theory]
    [MemberData(nameof(ReshapeTypeInferData))]
    public void TestReshapeTypeInfer(DistributedType inType, int[] newShape, IRType except)
    {
        var reshape = IR.F.Tensors.Reshape(new Var(inType), newShape);
        if (except is InvalidType)
        {
            Assert.IsType<InvalidType>(reshape.CheckedType);
        }
        else
        {
            Assert.Equal(except, reshape.CheckedType);
        }
    }
}
