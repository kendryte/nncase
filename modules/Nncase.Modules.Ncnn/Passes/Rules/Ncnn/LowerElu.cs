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
public partial class LowerElu : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsElu(
      IsWildcard("input") with { TypePattern = IsFloat() & HasFixedShape() },
      IsTensorConst("alpha") with { TypePattern = IsFloatScalar() });

    private Expr? GetReplace(Expr input, float alpha)
    {
        var newInput = new Var(input.CheckedType);

        return new Call(new Fusion("ncnn", NcnnElu(newInput, alpha), new[] { newInput }), input);
    }
}
