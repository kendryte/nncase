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
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
///
///  input     const
///    |     /
///    mul()   const
///    |     /
///    add()
///   |
/// conv2d(,w,bias)
/// ---------------------
///   input
///   |   mul(w,mulConst)
///   |    /
///   |   |
///   |   |
///   |   |  conv2d(addConst,w,zero_bias)
///   |  |   /
/// conv2d( ).
/// <remarks>
///  input can't be conv2d.
/// </remarks>
/// </summary>
[RuleGenerator]
public sealed partial class FoldConv2DAddMul : RewriteRule<CallPattern>
{
    private static readonly Pattern _mulConst = IsTensorConst("mulConst", c => CheckConstTensor(c.Value));

    private static readonly Pattern _addConst = IsTensorConst("addConst", c => CheckConstTensor(c.Value));

    private static readonly Pattern _inputPattern = IsWildcard("input", x => x is not Const);

    private static readonly Pattern _mulPattern =
      IsAlt(
        IsBinary("mul", "mulCall", op => op.BinaryOp == BinaryOp.Mul, _mulConst, _inputPattern),
        IsBinary("mul", "mulCall", op => op.BinaryOp == BinaryOp.Mul, _inputPattern, _mulConst));

    private static readonly Pattern _addPattern =
      IsAlt(
        IsBinary("add", "addCall", op => op.BinaryOp == BinaryOp.Add, _addConst, _mulPattern),
        IsBinary("add", "addCall", op => op.BinaryOp == BinaryOp.Add, _mulPattern, _addConst));

    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsConv2D(
        "conv2d",
        "conv2dCall",
        op => true,
        _addPattern,
        IsTensorConst("weights"),
        IsTensorConst("bias"),
        IsWildcard("strides"),
        IsWildcard("paddings"),
        IsWildcard("dilation"),
        IsWildcard("groups"),
        IsWildcard("fusedClamp"));

    private static bool CheckConstTensor(Tensor t)
    {
        if (t.ElementType != DataTypes.Float32)
        {
            return false;
        }

        if (!(t.Rank == 1 ||
             (t.Rank == 4 && t.Shape[0].FixedValue == 1 && t.Shape[2].FixedValue == 1 && t.Shape[3].FixedValue == 1) ||
             (t.Rank == 3 && t.Shape[1].FixedValue == 1 && t.Shape[2].FixedValue == 1)))
        {
            return false;
        }

        return true;
    }

    private Expr? GetReplace(Call conv2dCall, IR.NN.Conv2D conv2d, Tensor<float> weights, Tensor<float> bias, Expr strides, Expr paddings, Expr dilation, Expr groups, Expr fusedClamp, Tensor<float> addConst, Tensor<float> mulConst, Expr input)
    {
        int ic = weights.Shape[1].FixedValue;
        if (mulConst.Length != ic || addConst.Length != ic)
        {
            return null;
        }

        if (weights.Shape[2].FixedValue != 1 && weights.Shape[3].FixedValue != 1)
        {
            return null;
        }

        var newWeights = IR.F.Math.Mul(weights, Reshape(mulConst, new[] { 1, ic, 1, 1 }));

        var addConv = Conv2D(Reshape(addConst, new[] { 1, ic, 1, 1 }), weights, Tensor.FromScalar<float>(0.0f, weights.Shape[0].FixedValue), strides, paddings, dilation, conv2d.PadMode, groups, new float[]
        {
          ValueRange<float>.Full.Min,
          ValueRange<float>.Full.Max,
        });

        var addBias = addConv + Reshape(bias, new[] { 1, bias.Shape[0].FixedValue, 1, 1 });

        return Conv2D(input, newWeights, Reshape(addBias, new[] { bias.Shape[0].FixedValue }), strides, paddings, dilation, conv2d.PadMode, groups, fusedClamp);
    }
}
