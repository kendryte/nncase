// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// squeeze to reshape.
/// </summary>
[RuleGenerator]
public sealed partial class UnSqueezeToReshape : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsUnsqueeze(
        IsWildcard("input"),
        IsFixedShape("axes"));

    private Expr? GetReplace(Expr input, long[] axes)
    {
        var outputRank = input.CheckedShape.Rank + axes.Length;
        axes = axes.Select(a => a >= 0 ? a : outputRank + a).ToArray();
        var newShape = Array.Empty<Dimension>().ToList();
        var oldShape = input.CheckedShape.ToArray();
        var count = 0;
        for (var i = 0; i < outputRank; i++)
        {
            if (axes.Contains(i))
            {
                newShape.Add(1);
            }
            else
            {
                newShape.Add(oldShape[count++]);
            }
        }

        return Reshape(input, new RankedShape(newShape));
    }
}
