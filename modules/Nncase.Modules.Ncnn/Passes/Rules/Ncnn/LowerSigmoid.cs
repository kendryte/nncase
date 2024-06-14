// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.PatternMatch;

using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerSigmoid : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsSigmoid(
      IsWildcard("input") with { TypePattern = HasFixedShape() });

    private Expr? GetReplace(Expr input)
    {
        if (input.CheckedShape.Rank > 4 || (input.CheckedShape.Rank == 4 && input.CheckedShape[0].FixedValue != 1))
        {
            return null;
        }

        var newInput = input;
        var newInputVar = new Var(newInput.CheckedType);

        if (input.CheckedShape.Count == 4 && input.CheckedShape[0].FixedValue == 1)
        {
            newInput = Squeeze(input, new[] { 0 });
            newInputVar = new Var(newInput.CheckedType);
        }

        var sigmoid = new Call(new Fusion("ncnn", NcnnSigmoid(newInputVar), new[] { newInputVar }), newInput);

        if (input.CheckedShape.Count == 4 && input.CheckedShape[0].FixedValue == 1)
        {
            return Unsqueeze(sigmoid, new int[] { 0 });
        }
        else
        {
            return sigmoid;
        }
    }
}
