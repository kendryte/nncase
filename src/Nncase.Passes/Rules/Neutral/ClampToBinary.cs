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
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// convert <see cref="IR.NN.Relu"/> to <see cref="IR.Math.Clamp"/>.
/// </summary>
[RuleGenerator]
public sealed partial class ClampToBinary : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsClamp(
      IsWildcard("input"),
      IsTensorConst("min", t => t.Value.ElementType == DataTypes.Float32),
      IsTensorConst("max", t => t.Value.ElementType == DataTypes.Float32));

    private Expr? GetReplace(Expr input, Tensor<float> min, Tensor<float> max)
    {
        if (max.All(v => v >= float.MaxValue))
        {
            var distinct = min.Distinct().ToArray();
            if (distinct.Length == 1)
            {
                return IR.F.Math.Max(input, distinct[0]);
            }
            else
            {
                return IR.F.Math.Max(input, min);
            }
        }

        if (min.All(v => v <= float.MinValue))
        {
            var distinct = max.Distinct().ToArray();
            if (distinct.Length == 1)
            {
                return IR.F.Math.Min(input, distinct[0]);
            }
            else
            {
                return IR.F.Math.Min(input, min);
            }
        }

        return IR.F.Math.Max(IR.F.Math.Min(input, max), min);
    }
}
