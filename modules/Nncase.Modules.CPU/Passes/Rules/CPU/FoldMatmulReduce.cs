// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.CPU;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.CPU;

[RuleGenerator]
public sealed partial class FoldPackedMatmulReduce : IRewriteRule
{
    public IPattern Pattern { get; } =
        IsBoxing(
            target_name: "boxing",
            op => op.NewType is DistributedType dt && dt.NdSBP.All(s => s is not SBPPartial),
            IsPackedMatMul(
                "mm",
                "call",
                _ => true,
                IsWildcard("lhs"),
                IsWildcard("rhs")));

    public Expr? GetReplace(Call call, PackedMatMul mm, Expr lhs, Expr rhs)
    {
        if (call.CheckedType is DistributedType dt && dt.NdSBP.Any(s => s is SBPPartial))
        {
            var newMatmul = new IR.CPU.PackedMatMul(mm.LhsPackedAxes, mm.LhsPadedNums, mm.RhsPackedAxes, mm.RhsPadedNums, mm.TransposeA, mm.TransposeB, true);
            return new Call(newMatmul, lhs, rhs);
        }

        return null;
    }
}
