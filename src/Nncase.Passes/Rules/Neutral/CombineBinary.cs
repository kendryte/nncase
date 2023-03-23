// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// clamp(add(x,addConst),min,max) => add(clamp(x,min-addConst,max-addConst),addConst).
/// </summary>
[RuleGenerator]
public sealed partial class CombineClampAdd : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; init; } = IsClamp(
      CombineClampUtility.GetBinaryPattern(BinaryOp.Add, "add"),
      IsTensorConst("min", t => t.Value.ElementType == DataTypes.Float32),
      IsTensorConst("max", t => t.Value.ElementType == DataTypes.Float32));

    private Expr? GetReplace(Expr input, Tensor<float> addConst, Tensor<float> min, Tensor<float> max)
    {
        var newType = Evaluator.TypeInference.BroadcastType(new TensorType(addConst.ElementType, addConst.Shape), new TensorType(min.ElementType, min.Shape));
        if (newType is not TensorType newClampType || !newClampType.Shape.IsFixed)
        {
            return null;
        }

        var newMin = (min.ToOrtTensor() - addConst.ToOrtTensor()).ToValue();
        var newMax = (max.ToOrtTensor() - addConst.ToOrtTensor()).ToValue();
        return Add(Clamp(input, Const.FromValue(newMin), Const.FromValue(newMax)), addConst);
    }
}

/// <summary>
/// clamp(mul(x,mulConst),min,max) => mul(clamp(x,min/mulConst,max/mulConst),mulConst).
/// </summary>
[RuleGenerator]
public sealed partial class CombineClampMul : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsClamp(
      CombineClampUtility.GetBinaryPattern(BinaryOp.Mul, "mul"),
      IsTensorConst("min", t => t.Value.ElementType == DataTypes.Float32),
      IsTensorConst("max", t => t.Value.ElementType == DataTypes.Float32));

    private Expr? GetReplace(Expr input, Tensor<float> mulConst, Tensor<float> min, Tensor<float> max)
    {
        var newType = Evaluator.TypeInference.BroadcastType(new TensorType(mulConst.ElementType, mulConst.Shape), new TensorType(min.ElementType, min.Shape));
        if (newType is not TensorType newClampType || !newClampType.Shape.IsFixed)
        {
            return null;
        }

        // avoid div zero.
        // if (mulConst.All(f => f == 0.0f))
        // {
        //     return null;
        // }
        var tmulConst = mulConst.ToOrtTensor();
        var mint = min.ToOrtTensor();
        var maxt = max.ToOrtTensor();
        var cond = OrtKISharp.OrtKI.Greater(tmulConst, OrtKISharp.Tensor.FromScalar<float>(0.0f));
        var tmpMin = mint / tmulConst;
        var tmpMax = maxt / tmulConst;
        var newMin = OrtKISharp.OrtKI.Where(cond, tmpMin, tmpMax);
        var newMax = OrtKISharp.OrtKI.Where(cond, tmpMax, tmpMin);
        return Mul(
          Clamp(
            input,
            Const.FromValue(newMin.ToValue()),
            Const.FromValue(newMax.ToValue())),
          mulConst);
    }
}

internal static class CombineClampUtility
{
    public static Pattern GetInputPattern() => IsWildcard("input", x => x is not Const) with { TypePattern = HasDataType(DataTypes.Float32) };

    public static Pattern GetConstPattern(string prefix) => IsTensorConst(prefix + "Const", t => t.Value.ElementType == DataTypes.Float32);

    public static Pattern GetBinaryPattern(BinaryOp op, string constPrefix) => IsAlt(
      IsBinary(op, GetInputPattern(), GetConstPattern(constPrefix)),
      IsBinary(op, GetConstPattern(constPrefix), GetInputPattern()));
}
