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
            return IR.F.CPU.Boxing(input.Value, type);
        }

        return null;
    }
}

[RuleGenerator]
public partial class SplitPartialAndReshardBoxing : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsBoxing(
        target_name: "boxing",
        call_name: "call",
        _ => true,
        IsWildcard("input"));

    private Expr? GetReplace(Call call, Expr input)
    {
        if (input.CheckedType is DistributedType it && it.NdSBP.Any(sbp => sbp is SBPPartial) && call.CheckedType is DistributedType ot)
        {
            var newSBPs = it.NdSBP.Select(sbp => sbp is SBPPartial ? SBP.B : sbp).ToArray();
            if (newSBPs.Length != ot.NdSBP.Count || Enumerable.Range(0, newSBPs.Length).Any(i => newSBPs[i] != ot.NdSBP[i]))
            {
                return IR.F.CPU.Boxing(IR.F.CPU.Boxing(input, new DistributedType(it.TensorType, newSBPs, it.Placement)), ot);
            }

            return null;
        }

        return null;
    }
}
