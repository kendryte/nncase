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
            op => op.NewType is DistributedType dt && dt.NdSBP.All(s => s != SBP.P),
            IsPackedMatMul(
                "mm",
                "call",
                _ => true,
                IsWildcard("lhs"),
                IsWildcard("rhs")));

    public Expr? GetReplace(Call call, PackedMatMul mm, Expr lhs, Expr rhs)
    {
        if (call.CheckedType is DistributedType dt && dt.NdSBP.Any(s => s == SBP.P))
        {
            var newMatmul = new IR.CPU.PackedMatMul(mm.LhsPackedAxes, mm.LhsPadedNums, mm.RhsPackedAxes, mm.RhsPadedNums, mm.TransposeA, mm.TransposeB, true);
            return new Call(newMatmul, lhs, rhs);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class SwapUnpackReduce : IRewriteRule
{
    public IPattern Pattern { get; } =
        IsBoxing(
            target_name: "boxing",
            op => op.NewType is DistributedType dt && dt.NdSBP.All(s => s != SBP.P),
            IsUnpack(
                target_name: "unpack",
                _ => true,
                IsPackedMatMul(
                    "mm",
                    "call",
                    _ => true,
                    IsWildcard("lhs"),
                    IsWildcard("rhs"))));

    public Expr? GetReplace(Call call, Boxing boxing, Unpack unpack)
    {
        if (call.CheckedType is DistributedType dt && dt.NdSBP.Any(s => s == SBP.P))
        {
            var newType = new DistributedType(dt.TensorType, dt.NdSBP.Select(s => s is SBPPartialSum ? SBP.B : s).ToArray(), dt.Placement);
            var newBoxing = IR.F.CPU.Boxing(call, newType, boxing.IsReshape);
            return IR.F.CPU.Unpack(newBoxing, [.. unpack.Lanes], [.. unpack.Axes]);
        }

        return null;
    }
}
