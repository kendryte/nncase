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
public partial class LowerInstanceNorm : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsInstanceNormalization(
      IsWildcard("input"),
      IsTensorConst("gamma"),
      IsTensorConst("beta"),
      IsTensorConst("epsilon") with { TypePattern = IsFloatScalar() });

    private Expr? GetReplace(Expr input, float[] gamma, float[] beta, float epsilon)
    {
        // TODO: split input
        if (input.CheckedShape.ToList()[0] != 1)
        {
            return null;
        }

        var inRes = Squeeze(input, new[] { 0 });
        var inResO = new Var(inRes.CheckedType);

        int affine = gamma.All(a => a != 1) && beta.All(a => a != 0) ? 1 : 0;

        var instanceNorm = new Call(new Fusion("ncnn", NcnnInstanceNorm(inResO, gamma.Length, epsilon, affine, gamma, beta), new[] { inResO }), inRes);

        return Unsqueeze(instanceNorm, new[] { 0 });
    }
}
