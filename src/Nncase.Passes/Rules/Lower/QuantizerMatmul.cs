// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.Passes;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Lower;

[RuleGenerator]
public sealed partial class QuantizerMatmul : IRewriteRule
{
    public IPattern Pattern { get; }
        = IsRangeOfMarker(
            "markerC",
            IsMatMul(
                "matmul",
                "call",
                _ => true,
                IsRangeOfMarker("markerA", IsWildcard("inputA"), IsTensorConst("scaleA")),
                IsRangeOfMarker("markerB", IsWildcard("inputB"), IsTensorConst("scaleB"))),
            IsWildcard("scaleC"));

    private Expr? GetReplace(Expr matmul, Call call, Expr inputA, Marker markerA, TensorConst scaleA, Expr inputB, Marker markerB, TensorConst scaleB, Marker markerC, RunPassContext context)
    {
        if (markerA.MixQuantInfo!.MarkerQuantType != DataTypes.Float8 || markerB.MixQuantInfo!.MarkerQuantType != DataTypes.Float8)
        {
            return null;
        }

        if (inputA is not TensorConst && inputB is not TensorConst)
        {
            return null;
        }

        if (inputA is TensorConst)
        {
            var debug = 0;
        }
        else
        {
            var deqScaleA = scaleA.Value.ToScalar<float>();
            var deqScaleB = scaleB.Value.ToScalar<float>();
            var qScaleA = 1 / deqScaleA;
            var qScaleB = 1 / deqScaleB;
            var qInput = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, inputA, qScaleA);
            qInput = Nncase.IR.F.Tensors.Cast(qInput, DataTypes.Float8);
            var weights = ((TensorConst)inputB).Value.ToArray<float>();
            var qWeights = new Float8[weights.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                qWeights[i] = (Float8)(weights[i] * qScaleB);
            }

            var qWeightsConst = Tensor.From<Float8>(qWeights, inputB.CheckedShape.ToValueArray());
            var qMatmul = Nncase.IR.F.Math.MatMul(qInput, qWeightsConst);
            qMatmul = Nncase.IR.F.Tensors.Cast(qMatmul, DataTypes.Float32);
            qMatmul = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, qMatmul, deqScaleA * deqScaleB);
            CompilerServices.DumpIR(qMatmul, "qMatmul", "/compiler3.0/dev3.0/nncase/tests_output/test_debug");
            return qMatmul;
        }

        if (inputB is TensorConst)
        {
            var debug = 0;
        }
        else
        {
            var debug = 0;
        }


        return null;
    }
}
