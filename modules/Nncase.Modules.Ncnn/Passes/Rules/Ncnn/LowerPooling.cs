// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.PatternMatch;
using Nncase.ArgsStruct;

using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;
using Nncase.IR.NN;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerPooling : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsReduceWindow2D(
        "pdp",
        _ => true,
        IsWildcard("input"),
        IsWildcard("initValue"),
        IsWildcard("filter"),
        IsWildcard("stride"),
        IsTensorConst("padding"),
        IsWildcard("dilation"),
        IsWildcard("ceilMode"),
        IsWildcard("countIncludePad"));

    private Expr? GetReplace(ReduceWindow2D pdp, Expr input, float initValue, Expr filter, Expr stride, Expr padding, Expr dilation, bool ceilMode, bool countIncludePad)
    {
        // TODO: split input
        if (input.CheckedShape.ToList()[0] != 1)
        {
            return null;
        }

        var poolingType = pdp.ReduceOp switch
        {
            ReduceOp.Max => 0,
            ReduceOp.Mean => 1,
            _ => throw new NotImplementedException($"{pdp.ReduceOp} not suppor in ncnn!"),
        };

        var kernel_ = filter.Evaluate().AsTensor().ToArray<int>();
        var (kernelW, kernelH) = (kernel_[1], kernel_[0]);

        var stride_ = stride.Evaluate().AsTensor().ToArray<int>();
        var (strideW, strideH) = (stride_[1], stride_[0]);

        var padding_ = padding.Evaluate().AsTensor().ToArray<int>();
        var (padLeft, padTop, padRight, padBottom) = (padding_[2], padding_[3], padding_[0], padding_[1]);

        // Globalpool has been converted to pool, reflected in the size of the kernel.
        var args = new PoolingArgs(poolingType, kernelW, kernelH, strideW, strideH, padLeft, padRight, padTop, padBottom, false, ceilMode ? 0 : 1, countIncludePad, false, 0, 0, ceilMode);

        var inRes = Squeeze(input, new[] { 0 });
        var inResO = new Var(inRes.CheckedType);

        var pool = new Call(new Fusion("ncnn", NcnnPooling(inResO, args), new[] { inResO }), inRes);
        return Unsqueeze(pool, new[] { 0 });
    }
}
