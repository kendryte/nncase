// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Fold nop <see cref="IR.Tensors.Concat"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldPrePostReshapeSoftmax : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsReshape(
        "reshape",
        "reshapeCall",
        _ => true,
        IsSoftmax("softmax", IsReshape("rehsape2", "reshapeCall2", _ => true, IsWildcard("input"), IsTensorConst("shape2"))),
        IsTensorConst("shape1"));

    private Expr? GetReplace(Expr input)
    {
        return Softmax(input, 3);
    }
}
