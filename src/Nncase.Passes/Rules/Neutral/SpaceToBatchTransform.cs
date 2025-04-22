// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// squeeze to reshape.
/// </summary>
[RuleGenerator]
public sealed partial class SpaceToBatchToPad : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsSpaceToBatch(
        IsWildcard("input") with { TypePattern = HasFixedShape() },
        IsShape("blockShape"),
        IsPaddings("paddings"));

    private Expr? GetReplace(Expr input, Shape blockShape, Paddings paddings)
    {
        if (input.CheckedShape.Rank == 4 && blockShape.Rank == 2 && blockShape[0] == 1 && blockShape[1] == 1)
        {
            var newPaddings = new Padding[4];

            // pad for hw
            for (var i = 0; i < paddings.Rank; i++)
            {
                newPaddings[i] = (0, paddings[i].Before);
            }

            return Pad(input, newPaddings, PadMode.Constant, 0f);
        }

        return null;
    }
}
