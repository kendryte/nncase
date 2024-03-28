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
public partial class LowerLRN : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsLRN(
      IsWildcard("input"),
      IsTensorConst("alpha") with { TypePattern = IsScalar() },
      IsTensorConst("beta") with { TypePattern = IsScalar() },
      IsTensorConst("bias") with { TypePattern = IsScalar() },
      IsTensorConst("size") with { TypePattern = IsIntegralScalar() });

    private Expr? GetReplace(Expr input, float alpha, float beta, float bias, int size)
    {
        // TODO: split input
        if (input.CheckedShape.ToList()[0] != 1)
        {
            return null;
        }

        var inRes = Squeeze(input, new[] { 0 });
        var inResO = new Var(inRes.CheckedType);

        var lrn = new Call(new Fusion("ncnn", NcnnLRN(inResO, alpha, beta, bias, size), new[] { inResO }), inRes);

        return Unsqueeze(lrn, new[] { 0 });
    }
}
