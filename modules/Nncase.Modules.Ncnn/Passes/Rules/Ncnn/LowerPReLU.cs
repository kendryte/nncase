﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.IR.NN;
using Nncase.PatternMatch;

using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerPReLU : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsPRelu(
        IsWildcard("input") with { TypePattern = HasFixedShape() },
        IsTensorConst("slope"));

    private Expr? GetReplace(Expr input, Tensor<float> slope)
    {
        // TODO: split input
        if (input.CheckedShape.ToList()[0] != 1 || input.CheckedShape.Rank > 4)
        {
            return null;
        }

        var inRes = Squeeze(input, new[] { 0 });
        var inResO = new Var(inRes.CheckedType);

        var pReLU = new Call(new Fusion("ncnn", NcnnPReLU(inResO, slope.ToArray()), new[] { inResO }), inRes);
        return Unsqueeze(pReLU, new[] { 0 });
    }
}
