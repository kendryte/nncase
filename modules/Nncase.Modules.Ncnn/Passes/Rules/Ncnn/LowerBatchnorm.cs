// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerBatchNorm : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsBatchNormalization(
        IsWildcard("input") with { TypePattern = HasFixedShape() },
        IsTensorConst("scale"),
        IsTensorConst("bias"),
        IsTensorConst("mean"),
        IsTensorConst("var"),
        IsTensorConst("eps"));

    private Expr? GetReplace(Expr input, Tensor<float> scale, Tensor<float> bias, Tensor<float> mean, Tensor<float> var, Tensor<float> eps)
    {
        if (input.CheckedShape.Rank <= 1)
        {
            return null;
        }

        var shape = input.CheckedShape.ToValueArray().Reverse().Take(input.CheckedShape.Rank - 1).Reverse().ToArray();

        var newVar = Add(var, eps).Evaluate().AsTensor().ToArray<float>();


        var inRes = Reshape(input, shape);
        var inResO = new Var(inRes.CheckedType);

        var ncnnBatchNorm = new Call(new Fusion("ncnn", NcnnBatchNorm(inResO, shape[0], 0.0f, scale.ToArray(), mean.ToArray(), newVar.ToArray(), bias.ToArray()), new[] { inResO }), inRes);

        var outRes = Reshape(ncnnBatchNorm, input.CheckedShape);
        return outRes;
    }
}
