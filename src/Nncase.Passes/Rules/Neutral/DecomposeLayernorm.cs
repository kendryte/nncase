// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Decompose layernorm.
/// </summary>
[RuleGenerator]
public sealed partial class DecomposeLayerNorm : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
    IsLayerNorm(
      "ln",
      "call",
      _ => true,
      IsWildcard("input") with { TypePattern = IsFloat() },
      IsWildcard("scale") with { TypePattern = IsFloat() },
      IsWildcard("bias") with { TypePattern = IsFloat() });

    private Expr? GetReplace(Expr input, Call call, LayerNorm ln, TensorConst scale, TensorConst bias)
    {
        // TODO: only support ChannelLast now
        if (ln.Axis == -1 || ln.Axis == call.CheckedShape.Rank - 1)
        {
            var normalizedAxis = ln.Axis < 0 ? ln.Axis + input.CheckedShape.Rank : ln.Axis;
            if (ln.UseMean)
            {
                var mean = IR.F.Tensors.ReduceMean(input, new[] { normalizedAxis }, 0f, true);
                var sub = IR.F.Math.Sub(input, mean);
                var sigma = IR.F.Tensors.ReduceMean(IR.F.Math.Square(sub), new[] { normalizedAxis }, 0f, true);
                var rsigma = IR.F.Math.Rsqrt(IR.F.Math.Add(sigma, Tensor.From<float>(new[] { ln.Epsilon }, [1])));
                return IR.F.Math.Add(IR.F.Math.Mul(IR.F.Math.Mul(sub, rsigma), scale), bias);
            }
            else
            {
                var sigma = IR.F.Tensors.ReduceMean(IR.F.Math.Square(input), new[] { normalizedAxis }, 0f, true);
                var rsigma = IR.F.Math.Rsqrt(IR.F.Math.Add(sigma, Tensor.From<float>(new[] { ln.Epsilon }, [1])));
                return IR.F.Math.Add(IR.F.Math.Mul(IR.F.Math.Mul(input, rsigma), scale), bias);
            }
        }

        return null;
    }
}
