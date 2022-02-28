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
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Transform.Rules;

/// <summary>
/// Fold call of constants.
/// </summary>
[RuleGenerator]
public partial class FoldConstCall : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsCall("call", IsWildcard(), IsVArgsRepeat(() => IsConst()));

    Const GetReplace(Call call)
    {
        return Const.FromValue(call.Evaluate());
    }
}

/// <summary>
/// Fold shape of.
/// </summary>
[RuleGenerator]
public partial class FoldShapeOf : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsShapeOf(null, "call", IsWildcard("wc")) with { TypePattern = IsTensor() };

    Const GetReplace(Call call)
    {
        return Const.FromShape(call.CheckedShape);
    }
}
