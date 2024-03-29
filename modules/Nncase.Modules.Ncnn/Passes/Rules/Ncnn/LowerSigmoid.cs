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
        if (input.CheckedShape.Count > 3)
        {
            var inRes = Squeeze(input, new[] { 0 });
            var inResO = new Var(inRes.CheckedType);

            var sigmoid = new Call(new Fusion("ncnn", NcnnSigmoid(inResO), new[] { inResO }), inRes);
            return Unsqueeze(sigmoid, new[] { 0 });
        }

        // if input has shape [1]
        var newInput = new Var(input.CheckedType);
        return new Call(new Fusion("ncnn", NcnnSigmoid(newInput), new[] { newInput }), input);
    }
}
