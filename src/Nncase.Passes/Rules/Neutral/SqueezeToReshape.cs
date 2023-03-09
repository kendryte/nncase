// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// squeeze to reshape.
/// </summary>
[RuleGenerator]
public sealed partial class SqueezeToReshape : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsSqueeze(
        IsWildcard("input") with { TypePattern = HasFixedShape() },
        IsTensorConst("axes"));

    private Expr? GetReplace(Expr input, TensorConst axes)
    {
        var axesArray = axes.Value.ToArray<long>().Select(a => a >= 0 ? a : input.CheckedShape.Count + a).ToArray();
        var newShape = axesArray.Length == 0
            ? input.CheckedShape.Where(d => !d.IsFixed || d.FixedValue != 1)
            : input.CheckedShape.Where((_, i) => !axesArray.Contains(i));
        return Reshape(input, new Shape(newShape.ToArray()));
    }
}
