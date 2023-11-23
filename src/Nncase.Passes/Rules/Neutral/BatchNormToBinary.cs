// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class BatchNormToBinary : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsBatchNormalization(
        "bn",
        "bnCall",
        _ => true,
        IsWildcard("input") with { TypePattern = HasFixedShape() },
        IsTensorConst("gamma"),
        IsTensorConst("beta"),
        IsTensorConst("mean"),
        IsTensorConst("var"),
        IsTensorConst("eps"));

    private Expr? GetReplace(Expr input, Tensor<float> gamma, Tensor<float> beta, Tensor<float> mean, Tensor<float> var, Tensor<float> eps, Expr bn, Call bnCall)
    {
        if (input.CheckedShape.Rank <= 1)
        {
            return null;
        }

        var shape = input.CheckedShape.ToValueArray();
        var bnShape = Enumerable.Repeat(1, shape.Length - 1).ToArray();
        bnShape[0] = shape[1];
        var scaleBn = IR.F.Math.Div(gamma, IR.F.Math.Sqrt(IR.F.Math.Add(var, eps))).With(metadata: new IRMetadata() { OutputNames = new[] { bnCall.Metadata.OutputNames?[0] + "_Scale" } });
        var biasBn = IR.F.Math.Sub(beta, IR.F.Math.Mul(gamma, IR.F.Math.Div(mean, IR.F.Math.Sqrt(IR.F.Math.Add(var, eps))))).With(metadata: new IRMetadata() { OutputNames = new[] { bnCall.Metadata.OutputNames?[0] + "_Bias" } });
        var mul = IR.F.Math.Mul(input, Reshape(scaleBn, bnShape).With(metadata: new IRMetadata() { OutputNames = new[] { bnCall.Metadata.OutputNames?[0] + "_Scale" } })).With(metadata: new IRMetadata() { OutputNames = new[] { bnCall.Metadata.OutputNames?[0] + "_BN_Mul" } });
        var binary = IR.F.Math.Add(mul, Reshape(biasBn, bnShape).With(metadata: new IRMetadata() { OutputNames = new[] { bnCall.Metadata.OutputNames?[0] + "_Bias" } })).With(metadata: new IRMetadata() { OutputNames = new[] { bnCall.Metadata.OutputNames?[0] + "_BN_Add" } });
        return binary;
    }
}
