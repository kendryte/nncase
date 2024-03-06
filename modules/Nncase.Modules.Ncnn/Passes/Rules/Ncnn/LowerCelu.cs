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
public partial class LowerCelu : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsCelu(
      IsWildcard("input") with { TypePattern = IsFloat() },
      IsTensorConst("alpha") with { TypePattern = IsFloatScalar() });

    private Expr? GetReplace(Expr input, float alpha)
    {
        if (input.CheckedShape.Count > 4 || input.CheckedShape[0].FixedValue != 1)
        {
            Console.WriteLine("ncnn not support more than 4D or batchSize > 1");
            return null;
        }

        var inRes = Squeeze(input, new[] { 0 });
        var inResO = new Var(inRes.CheckedType);
        var celu = new Call(new Fusion("ncnn", NcnnCelu(inResO, alpha), new[] { inResO }), inRes);
        return Unsqueeze(celu, new[] { 0 });
    }
}
