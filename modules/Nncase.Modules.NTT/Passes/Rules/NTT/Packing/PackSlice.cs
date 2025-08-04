// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
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
public sealed partial class VectorizeSlicePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
    PatternMatch.F.Tensors.IsVectorize(
            "vectorize",
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

    public static bool TryPropagateVectorize(ReadOnlySpan<int> vectorizeAxes, ReadOnlySpan<int> vectorizeLanes, RankedShape begins, RankedShape ends, RankedShape axes, RankedShape strides, [MaybeNullWhen(false)] out RankedShape newBegins, [MaybeNullWhen(false)] out RankedShape newEnds)
    {
        var newBeginValues = begins.ToArray();
        var newEndValues = ends.ToArray();

        for (var i = 0; i < vectorizeAxes.Length; i++)
        {
            var axis = vectorizeAxes[i];
            var lanes = vectorizeLanes[i];

            var sliceAxisIndex = axes.IndexOf(axis);
            if (sliceAxisIndex != -1)
            {
                if (strides[sliceAxisIndex] == 1
                    && Dimension.TryDivExactly(newBeginValues[sliceAxisIndex], lanes, out var newBegin)
                    && Dimension.TryDivExactly(newEndValues[sliceAxisIndex], lanes, out var newEnd))
                {
                    // If the slice is aligned with the vectorize lanes, we can adjust the begins and ends
                    // to reflect the vectorize.
                    newBeginValues[sliceAxisIndex] = newBegin;
                    newEndValues[sliceAxisIndex] = newEnd;
                }
                else
                {
                    // If the slice is not aligned with the vectorize lanes, we cannot replace it.
                    newBegins = null;
                    newEnds = null;
                    return false;
                }
            }
        }

        newBegins = new RankedShape(newBeginValues);
        newEnds = new RankedShape(newEndValues);
        return true;
    }

    private Expr? GetReplace(Vectorize vectorize, Call caller, Call callee, Expr input, RankedShape begins, RankedShape ends, RankedShape axes, RankedShape strides)
    {
        if (TryPropagateVectorize(vectorize.Axes, vectorize.Lanes, begins, ends, axes, strides, out var newBegins, out var newEnds))
        {
            return callee.WithArguments([
                (Slice.Input, caller.WithArguments([(Vectorize.Input, input)])),
                (Slice.Begins, newBegins),
                (Slice.Ends, newEnds),
            ]);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class SliceDevectorizePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
    IsSlice(
        "slice",
        "caller",
        PatternMatch.F.Tensors.IsDevectorize(
            "devectorize",
            "callee",
            _ => true,
            IsWildcard("input")),
        IsRankedShape("begins"),
        IsRankedShape("ends"),
        IsFixedShape("axes"),
        IsFixedShape("strides"));

    private Expr? GetReplace(Devectorize devectorize, Call caller, Call callee, Expr input, RankedShape begins, RankedShape ends, RankedShape axes, RankedShape strides)
    {
        if (VectorizeSlicePropagation.TryPropagateVectorize(devectorize.Axes, devectorize.Lanes, begins, ends, axes, strides, out var newBegins, out var newEnds))
        {
            return callee.WithArguments([
                (Devectorize.Input, caller.WithArguments([
                    (Slice.Input, input),
                    (Slice.Begins, newBegins),
                    (Slice.Ends, newEnds),
                ])),
            ]);
        }

        return null;
    }
}
