// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Numerics.Tensors;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Transform.Rules.Neutral;

/// <summary>
/// Fold nop <see cref="IR.NN.Pad"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldNopPad : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsPad(padMode => true, IsWildcard("input"), IsTensorConst("pads", IsIntegral()), IsWildcard());

    private Expr? GetReplace(Expr input, TensorConst pads)
    {
        if (pads.Value.Cast<int>().All(x => x == 0))
        {
            return input;
        }

        return null;
    }
}

/// <summary>
/// Fold two <see cref="IR.NN.Pad"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldTwoPads : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsPad(
            PadMode.Constant,
            IsPad(PadMode.Constant, IsWildcard("input"), IsTensorConst("pads1", IsIntegral()), IsWildcard("padValue1")),
            IsTensorConst("pads2", IsIntegral()),
            IsWildcard("padValue2"));

    private Expr? GetReplace(Expr input, TensorConst pads1, TensorConst pads2, Expr padValue1, Expr padValue2)
    {
        if (padValue1.Equals(padValue2))
        {
            var (t1, t2) = (pads1.Value.Cast<int>(), pads2.Value.Cast<int>());
            var newt = new Tensor<int>(t1.Dimensions);
            for (int i = 0; i < t1.Dimensions[0]; i++)
            {
                newt[i, 0] = t1[i, 0] + t2[i, 0];
                newt[i, 1] = t1[i, 1] + t2[i, 1];
            }

            return Pad(input, newt, PadMode.Constant, padValue1);
        }

        return null;
    }
}
