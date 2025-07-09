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
public sealed partial class PackSlicePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
    PatternMatch.F.Tensors.IsPack(
            "pack",
            "caller",
            _ => true,
            IsSlice(
                "slice",
                "callee",
                IsWildcard("input"),
                IsRankedShape("begins"),
                IsRankedShape("ends"),
                IsFixedShape("axes"),
                IsFixedShape("strides")));

    private Expr? GetReplace(Pack pack, Call caller, Call callee, Expr input, RankedShape begins, RankedShape ends, RankedShape axes, RankedShape strides)
    {
        var newBegins = begins.ToArray();
        var newEnds = ends.ToArray();

        for (var i = 0; i < pack.Axes.Count; i++)
        {
            var axis = pack.Axes[i];
            var lanes = pack.Lanes[i];

            var sliceAxisIndex = axes.IndexOf(axis);
            if (sliceAxisIndex != -1)
            {
                if (strides[sliceAxisIndex] == 1
                    && Dimension.TryDivExactly(newBegins[sliceAxisIndex], lanes, out var newBegin)
                    && Dimension.TryDivExactly(newEnds[sliceAxisIndex], lanes, out var newEnd))
                {
                    // If the slice is aligned with the pack lanes, we can adjust the begins and ends
                    // to reflect the packing.
                    newBegins[sliceAxisIndex] = newBegin;
                    newEnds[sliceAxisIndex] = newEnd;
                }
                else
                {
                    // If the slice is not aligned with the pack lanes, we cannot replace it.
                    return null;
                }
            }
        }

        return callee.WithArguments(
            [
                (Slice.Input, caller.WithArguments([(Pack.Input, input)])),
                (Slice.Begins, new RankedShape(newBegins)),
                (Slice.Ends, new RankedShape(newEnds)),
            ]);
    }
}
