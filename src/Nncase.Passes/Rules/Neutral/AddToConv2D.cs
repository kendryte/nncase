// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.Passes;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Transform <see cref="IR.F.Math.Add(Expr, Expr)"/> to <see cref="IR.NN.Conv2D"/>.
/// </summary>
[RuleGenerator]
public sealed partial class AddToConv2D : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = Add(
        IsWildcard("a") with { TypePattern = HasRank(4) },
        IsWildcard("b") with { TypePattern = HasRank(4) });

    private Expr? GetReplace(Expr a, Expr b)
    {
        var a_sp = a.CheckedShape;
        var b_sp = b.CheckedShape;
        if (a_sp == b_sp)
        {
            var channels = a_sp[1].FixedValue;
            var weights = new Tensor<float>(channels * 2 * channels);
            for (int i = 0; i < channels; i++)
            {
                weights[(2 * channels * i) + i] = 1.0f;
                weights[(2 * channels * i) + i + channels] = 1.0f;
            }

            var c = Concat(new IR.Tuple(a, b), 1);
            var con_weights = Const.FromTensor(weights.Reshape(new[] { channels, 2 * channels, 1, 1 }));

            return Conv2D(
                c,
                con_weights,
                bias: Tensor.FromScalar(0.0f, channels),
                stride: new[] { 1, 1 },
                padding: Tensor.From(new[] { 0, 0, 0, 0 }, new[] { 2, 2 }),
                dilation: new[] { 1, 1 },
                padMode: PadMode.Constant,
                groups: 1);
        }

        return null;
    }
}
