// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.Transform.Rules;

/// <summary>
/// Fold call of constants.
/// </summary>
public class FoldConstCall : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsCall(IsWildcard(), IsVArgsRepeat(() => IsConst()));

    /// <inheritdoc/>
    public override Expr? GetReplace(IMatchResult result)
    {
        var expr = result.Get(Pattern);
        return Const.FromValue(expr.Evaluate());
    }
}

/// <summary>
/// Fold shape of.
/// </summary>
public class FoldShapeOf : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = ShapeOf(IsWildcard()) with { TypePattern = IsTensor() };

    /// <inheritdoc/>
    public override Expr? GetReplace(IMatchResult result)
    {
        return Const.FromShape(result.Get(Pattern).CheckedShape);
    }
}
