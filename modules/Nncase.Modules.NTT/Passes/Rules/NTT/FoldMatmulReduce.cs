// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.NTT;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Distributed;
using static Nncase.PatternMatch.F.NTT;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

[RuleGenerator]
public sealed partial class FoldVectorizedMatmulReduce : IRewriteRule
{
    public IPattern Pattern { get; } =
        IsBoxing(
            target_name: "boxing",
            op => op.NewType is DistributedType dt && dt.AxisPolicies.All(s => s is not SBPPartial),
            IsVectorizedMatMul(
                "mm",
                "call",
                _ => true,
                IsWildcard("lhs"),
                IsWildcard("rhs")));

    public Expr? GetReplace(Call call, VectorizedMatMul mm, Expr lhs, Expr rhs)
    {
        if (call.CheckedType is DistributedType dt && dt.AxisPolicies.Any(s => s is SBPPartial))
        {
            var newMatmul = new IR.NTT.VectorizedMatMul(mm.OutputDataType, mm.LhsVectorizedAxes, mm.RhsVectorizedAxes, mm.TransposeA, mm.TransposeB, true);
            return new Call(newMatmul, lhs, rhs);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class SwapDevectorizeReduce : IRewriteRule
{
    public IPattern Pattern { get; } =
        IsBoxing(
            target_name: "boxing",
            op => op.NewType is DistributedType dt && dt.AxisPolicies.All(s => s is not SBPPartial),
            IsDevectorize(
                target_name: "devectorize",
                _ => true,
                IsVectorizedMatMul(
                    "mm",
                    "call",
                    _ => true,
                    IsWildcard("lhs"),
                    IsWildcard("rhs"))));

    public Expr? GetReplace(Call call, Boxing boxing, Devectorize devectorize)
    {
        if (call.CheckedType is DistributedType dt && dt.AxisPolicies.Any(s => s is SBPPartial))
        {
            var newType = new DistributedType(dt.TensorType, dt.AxisPolicies.Select(s => s is SBPPartial ? SBP.B : s).ToArray(), dt.Placement);
            var newBoxing = IR.F.Distributed.Boxing(call, newType);
            return IR.F.Tensors.Devectorize(newBoxing, [.. devectorize.Lanes], [.. devectorize.Axes]);
        }

        return null;
    }
}
