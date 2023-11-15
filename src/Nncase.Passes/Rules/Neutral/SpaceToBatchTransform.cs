// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
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
        IsTensorConst("blockShape"),
        IsTensorConst("paddings"));

    private Expr? GetReplace(Expr input, Tensor<int> blockShape, Tensor<int> paddings)
    {
        var blockShapeArray = blockShape.ToArray();
        var paddingsArray = paddings.ToArray();
        if (input.CheckedShape.Rank == 4 && blockShapeArray.Length == 2 && blockShapeArray[0] == 1 && blockShape[1] == 1)
        {
            var newPaddingsArray = new int[8];

            // pad for hw
            for (var i = 0; i < paddingsArray.Length; i++)
            {
                newPaddingsArray[i + 4] = paddingsArray[i];
            }

            var newPaddings = Tensor.From(newPaddingsArray, new[] { 4, 2 });

            return Pad(input, newPaddings, PadMode.Constant, 0f);
        }

        return null;
    }
}
