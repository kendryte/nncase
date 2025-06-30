// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Distributed;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules;

[RuleGenerator]
public partial class FoldBoxingConst : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsBoxing(
      target_name: "boxing",
      _ => true,
      IsTensorConst("input"));

    private Expr? GetReplace(Boxing boxing, Tensor input)
    {
        var type = (DistributedType)boxing.NewType;
        return new TensorConst(input, type.AxisPolices, type.Placement);
    }
}

[RuleGenerator]
public partial class UnfoldDistributedConst : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsTensorConst("input");

    private Expr? GetReplace(TensorConst input)
    {
        var type = input.CheckedType;
        if (type is DistributedType)
        {
            return IR.F.Distributed.Boxing(input.Value, type);
        }

        return null;
    }
}
