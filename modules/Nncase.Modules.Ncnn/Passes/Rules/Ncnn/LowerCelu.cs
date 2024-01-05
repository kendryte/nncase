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
      IsTensorConst("alpha") with { TypePattern = IsFloatScalar() } );

    private Expr? GetReplace(Expr input, float alpha)
    {
        if (alpha != 1.0)
        {
            return false;
        }

        var newInput = new Var(input.CheckedType);

        return new Call(new Fusion("ncnn", NcnnCelu(newInput, alpha), new[] { newInput }), input);

    }
}
