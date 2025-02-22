// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class FoldZeroConcat : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsConcat("concat", op => true, IsTuple(IsVArgsRepeat("inputs", inputs =>
    {
        var ps = new Pattern[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            ps[i] = IsWildcard(i.ToString()) with { TypePattern = HasRank() };
        }

        return ps;
    })));

    private Expr? GetReplace(IR.Tensors.Concat concat, IReadOnlyList<Expr> inputs)
    {
        bool changed = false;
        var newInputs = new List<Expr>();
        foreach (var input in inputs)
        {
            if (input.CheckedShape.Any(x => x.IsFixed && x.FixedValue == 0))
            {
                changed = true;
                continue;
            }
            else
            {
                newInputs.Add(input);
            }
        }

        if (newInputs.Count == 0)
        {
            var shape = new int[inputs[0].CheckedShape.Rank];
            var value = Tensor.FromScalar(0, shape).CastTo(inputs[0].CheckedDataType);
            return value;
        }
        else if (newInputs.Count == 1)
        {
            return newInputs[0];
        }
        else if (changed)
        {
            return IR.F.Tensors.Concat(new IR.Tuple(newInputs.ToArray()), concat.Axis);
        }

        return null;
    }
}
