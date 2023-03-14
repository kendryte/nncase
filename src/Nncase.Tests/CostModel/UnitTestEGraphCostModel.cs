// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System.Collections.Generic;
using System.Linq;
using Nncase;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.Passes;
using Xunit;

namespace Nncase.Tests.CostModelTest;

public sealed class UnitTestEGraphCostModel
{
    [Fact]
    public void TestEGraphExtractMinBy()
    {
        var a = new Call(new IR.Math.RangeOf());
        var b = (Const)new[] { 1, 2, 3 };
        var c = new Call(new IR.Math.Unary(UnaryOp.Abs));
        var d = new Call(new IR.Math.Clamp());

        var list = new Expr[] { a, b, c, d };

        var cost = new Dictionary<Expr, Cost>(ReferenceEqualityComparer.Instance)
        {
          { a,
            new() {
              [CostFactorNames.MemoryLoad] = 76380,
              [CostFactorNames.MemoryLoad] = 42336,
              [CostFactorNames.CPUCycles] = 532529182956,
            }
          },
          { b, Cost.Zero },
          { c,
            new() {
              [CostFactorNames.MemoryLoad] = 37940,
              [CostFactorNames.MemoryLoad] = 20840,
              [CostFactorNames.CPUCycles] = 266073472105,
            }
          },
          { d, new() {
              [CostFactorNames.MemoryLoad] = 38080,
              [CostFactorNames.MemoryLoad] = 20980,
              [CostFactorNames.CPUCycles] = 266073472105,
            }
          },
        };

        Assert.IsType<TensorConst>(list.OrderBy(e => e, EGraphExtractExtensions.ENodeTypeComparer.Instance).First());

        Assert.True(cost[b] < cost[c]);

        Assert.IsType<TensorConst>(list.OrderBy(e => e, EGraphExtractExtensions.ENodeTypeComparer.Instance).MinBy(e => cost[e]));
    }
}
