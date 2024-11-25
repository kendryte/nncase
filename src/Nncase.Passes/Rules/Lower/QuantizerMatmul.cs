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
        = IsMatMul(
                "matmul",
                "call",
                _ => true,
                IsRangeOfMarker("markerA", IsWildcard("inputA"), IsTensorConst("scaleA")),
                IsRangeOfMarker("markerB", IsWildcard("inputB"), IsTensorConst("scaleB")));

    private Expr? GetReplace(Expr matmul, Call call, Expr inputA, Marker markerA, TensorConst scaleA, Expr inputB, Marker markerB, TensorConst scaleB, RunPassContext context)
    {
        if (inputA is not TensorConst && inputB is not TensorConst)
        {
            return null;
        }

        if (markerA.MixQuantInfo!.MarkerQuantType == DataTypes.Float8E4M3 && markerB.MixQuantInfo!.MarkerQuantType == DataTypes.Float8E4M3)
        {
            if (inputA is TensorConst)
            {
                return QuantMatmulE4M3(inputB, scaleB, inputA, scaleA);
            }
            else
            {
                return QuantMatmulE4M3(inputA, scaleA, inputB, scaleB);
            }
        }
        else if (markerA.MixQuantInfo!.MarkerQuantType == DataTypes.Float8E5M2 && markerB.MixQuantInfo!.MarkerQuantType == DataTypes.Float8E5M2)
        {
            if (inputA is TensorConst)
            {
                return QuantMatmulE5M2(inputB, scaleB, inputA, scaleA);
            }
            else
            {
                return QuantMatmulE5M2(inputA, scaleA, inputB, scaleB);
            }
        }
        else if (markerA.MixQuantInfo!.MarkerQuantType == DataTypes.Int8 && markerB.MixQuantInfo!.MarkerQuantType == DataTypes.Int8)
        {
            if (inputA is TensorConst)
            {
                return QuantMatmulInt8(inputB, scaleB, inputA, scaleA);
            }
            else
            {
                return QuantMatmulInt8(inputA, scaleA, inputB, scaleB);
            }
        }
        else
        {
            return null;
        }
    }

    private Expr? QuantMatmulE4M3(Expr inputA, TensorConst scaleA, Expr inputB, TensorConst scaleB)
    {
        var deqScaleA = scaleA.Value.ToScalar<float>();
        var deqScaleB = scaleB.Value.ToScalar<float>();
        var qScaleA = 1 / deqScaleA;
        var qScaleB = 1 / deqScaleB;
        var qInput = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, inputA, qScaleA);
        qInput = Nncase.IR.F.Tensors.Cast(qInput, DataTypes.Float8E4M3);
        var weights = ((TensorConst)inputB).Value.ToArray<float>();
        var qWeights = new Float8E4M3[weights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            qWeights[i] = (Float8E4M3)(weights[i] * qScaleB);
        }

        var qWeightsConst = Tensor.From<Float8E4M3>(qWeights, inputB.CheckedShape.ToValueArray());
        var qMatmul = Nncase.IR.F.Math.MatMul(qInput, qWeightsConst);
        qMatmul = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, qMatmul, deqScaleA * deqScaleB);
        return qMatmul;
    }

    private Expr? QuantMatmulE5M2(Expr inputA, TensorConst scaleA, Expr inputB, TensorConst scaleB)
    {
        var deqScaleA = scaleA.Value.ToScalar<float>();
        var deqScaleB = scaleB.Value.ToScalar<float>();
        var qScaleA = 1 / deqScaleA;
        var qScaleB = 1 / deqScaleB;
        var qInput = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, inputA, qScaleA);
        qInput = Nncase.IR.F.Tensors.Cast(qInput, DataTypes.Float8E5M2);
        var weights = ((TensorConst)inputB).Value.ToArray<float>();
        var qWeights = new Float8E5M2[weights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            qWeights[i] = (Float8E5M2)(weights[i] * qScaleB);
        }

        var qWeightsConst = Tensor.From<Float8E5M2>(qWeights, inputB.CheckedShape.ToValueArray());
        var qMatmul = Nncase.IR.F.Math.MatMul(qInput, qWeightsConst);
        qMatmul = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, qMatmul, deqScaleA * deqScaleB);
        return qMatmul;
    }

    private Expr? QuantMatmulInt8(Expr inputA, TensorConst scaleA, Expr inputB, TensorConst scaleB)
    {
        var deqScaleA = scaleA.Value.ToScalar<float>();
        var deqScaleB = scaleB.Value.ToScalar<float>();
        var qScaleA = 1 / deqScaleA;
        var qScaleB = 1 / deqScaleB;
        var qInput = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, inputA, qScaleA);
        qInput = Nncase.IR.F.Tensors.Cast(qInput, DataTypes.Int8);
        var weights = ((TensorConst)inputB).Value.ToArray<float>();
        var qWeights = new sbyte[weights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            qWeights[i] = (sbyte)(weights[i] * qScaleB);
        }

        var qWeightsConst = Tensor.From<sbyte>(qWeights, inputB.CheckedShape.ToValueArray());
        var qMatmul = Nncase.IR.F.Math.MatMul(qInput, qWeightsConst);
        qMatmul = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, qMatmul, deqScaleA * deqScaleB);
        return qMatmul;
    }
}
