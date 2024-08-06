// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Nncase.ArgsStruct;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;

using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerGELU : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsGelu(
        IsWildcard("input") with { TypePattern = HasFixedShape() },
        IsTensorConst("alpha"));

    private Expr? GetReplace(Expr input, float alpha)
    {
        if (input.CheckedShape.Count > 4 || input.CheckedShape[0].FixedValue != 1)
        {
            Console.WriteLine("ncnn not support more than 4D or batchSize > 1");
            return null;
        }

        // TODO: support GELU with scale.
        var inRes = Squeeze(input, new[] { 0 });
        var inResO = new Var(inRes.CheckedType);
        if (Math.Abs(alpha - 1.0) > 1e-06)
        {
            return null;
        }

        var c = new Call(new Fusion("ncnn", NcnnGELU(inResO), new[] { inResO }), inRes);
        return Unsqueeze(c, new[] { 0 });
    }
}
