// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

[RuleGenerator]
public sealed partial class VectorizeGatherPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsPack(
            "vectorize",
            "caller",
            _ => true,
            IsGather(
                "gather",
                "callee",
                _ => true,
                IsWildcard("input"),
                IsWildcard("index") with { TypePattern = HasRankedShape() }));

    private Expr? GetReplace(Pack vectorize, Gather gather, Call caller, Call callee, Expr input, Expr index)
    {
        if (index.CheckedShape.Rank == 1 && !vectorize.Axes.Contains(gather.Axis))
        {
            // If the vectorize does not contain the gather axis, we directly vectorize the gather input.
            return callee.WithArguments(
                [(Gather.Input, caller.WithArguments([(Pack.Input, input)]))]);
        }

        return null;
    }
}
