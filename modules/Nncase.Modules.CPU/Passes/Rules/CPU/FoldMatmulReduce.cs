// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.IR.Distributed;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.CPU;
using static Nncase.PatternMatch.F.Distributed;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.CPU;

[RuleGenerator]
public sealed partial class FoldPackedMatmulReduce : IRewriteRule
{
    public IPattern Pattern { get; } =
        IsBoxing(
            target_name: "boxing",
            op => op.NewType is DistributedType dt && dt.AxisPolices.All(s => s is not SBPPartial),
            IsPackedMatMul(
                "mm",
                "call",
                _ => true,
                IsWildcard("lhs"),
                IsWildcard("rhs")));

    public Expr? GetReplace(Call call, PackedMatMul mm, Expr lhs, Expr rhs)
    {
        if (call.CheckedType is DistributedType dt && dt.AxisPolices.Any(s => s is SBPPartial))
        {
            var newMatmul = Nncase.IR.F.CPU.PackedMatMul(lhs, rhs, mm.LhsPackedAxes, mm.LhsPadedNums, mm.RhsPackedAxes, mm.RhsPadedNums, mm.TransposeA, mm.TransposeB, true, mm.OutputDataType);
            return newMatmul;
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
            op => op.NewType is DistributedType dt && dt.AxisPolices.All(s => s is not SBPPartial),
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
        if (call.CheckedType is DistributedType dt && dt.AxisPolices.Any(s => s is SBPPartial))
        {
            var newType = new DistributedType(dt.TensorType, dt.AxisPolices.Select(s => s is SBPPartial ? SBP.B : s).ToArray(), dt.Placement);
            var newBoxing = IR.F.Distributed.Boxing(call, newType);
            return IR.F.CPU.Unpack(newBoxing, [.. unpack.Lanes], [.. unpack.Axes]);
        }

        return null;
    }
}
