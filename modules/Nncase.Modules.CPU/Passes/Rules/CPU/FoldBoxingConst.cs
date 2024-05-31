// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.CPU;
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
        return new TensorConst(input, type.NdSBP, type.Placement);
    }
}
