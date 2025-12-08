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
public partial class UpdateBoxingTensorType : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsBoxing(
      target_name: "boxing",
      _ => true,
      IsWildcard("input"));

    private Expr? GetReplace(Boxing boxing, Expr input, RunPassContext context)
    {
        var type = input.CheckedType;
        if (type is DistributedType dt1 && boxing.NewType is DistributedType dt2)
        {
            var ttype = dt1.TensorType;
            var dtype = dt2 with { TensorType = ttype };
            var newBoxing = new Call(new IR.Distributed.Boxing(dtype), input);
            context.MatchOptions.SuppressPattern(newBoxing, Pattern);
            return newBoxing;
        }

        return null;
    }
}
