// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public partial class SimplifySelect : IRewriteRule
{

    private readonly Pattern _operandPattern = IsWildcard("operand");

    public SimplifySelect()
    {
        Pattern = IsSelect(
            "selectOp",
            "select",
            _ => true,
            IsWildcard("cond"),
            _operandPattern,
            _operandPattern);
    }

    public IPattern Pattern { get; }

    private Expr? GetReplace(Expr operand)
    {
        return operand;
    }
}
