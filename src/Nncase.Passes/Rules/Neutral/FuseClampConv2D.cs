// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
///
///         Conv2d
///           |            =>   Conv2d
///          clamp.
///
/// </summary>
[RuleGenerator]
public sealed partial class FuseClampConv2D : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } =
      IsClamp(
        IsCallSpecific("call", IsOp<Conv2D>("op"), (Conv2D.FusedClamp, IsTensorConst("fusedClamp"))),
        IsTensorConst("min", t => t.Value.Shape.IsScalar && t.Value.ElementType == DataTypes.Float32),
        IsTensorConst("max", t => t.Value.Shape.IsScalar && t.Value.ElementType == DataTypes.Float32));

    private Expr? GetReplace(Op op, IReadOnlyList<Expr> callParams, float[] fusedClamp, float min, float max)
    {
        var newClamp = new float[2];
        newClamp[0] = System.MathF.Max(min, fusedClamp[0]);
        newClamp[1] = System.MathF.Min(max, fusedClamp[1]);
        return ReplaceCallParams(op, callParams, (Conv2D.FusedClamp, newClamp));
    }
}
