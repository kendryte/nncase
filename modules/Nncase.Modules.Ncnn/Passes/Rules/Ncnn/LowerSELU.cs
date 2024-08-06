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
public partial class LowerSELU : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsSelu(
      IsWildcard("input") with { TypePattern = HasFixedShape() },
      IsTensorConst("alpha"),
      IsTensorConst("gamma"));

    private Expr? GetReplace(Expr input, float alpha, float gamma)
    {
        var inRes = Squeeze(input, new[] { 0 });
        var inResO = new Var(inRes.CheckedType);

        var selu = new Call(new Fusion("ncnn", NcnnSELU(inResO, alpha, gamma), new[] { inResO }), inRes);

        return Unsqueeze(selu, new[] { 0 });
    }
}
