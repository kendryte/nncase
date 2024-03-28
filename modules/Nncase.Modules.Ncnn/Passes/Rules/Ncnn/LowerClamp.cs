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
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerClamp : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsClamp(
      IsWildcard("input"),
      IsTensorConst("min", t => t.Value.ElementType == DataTypes.Float32),
      IsTensorConst("max", t => t.Value.ElementType == DataTypes.Float32));

    private Expr? GetReplace(Expr input, float min, float max)
    {
        var newInput = new Var(input.CheckedType);

        return new Call(new Fusion("ncnn", NcnnClip(newInput, min, max), new[] { newInput }), input);
    }
}
