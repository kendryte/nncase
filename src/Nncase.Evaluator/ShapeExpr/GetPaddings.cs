// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.ShapeExpr;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using static Nncase.IR.F.Tensors;

namespace Nncase.Evaluator.ShapeExpr;

public partial class GetPaddingsEvaluator : IEvaluator<GetPaddings>, ITypeInferencer<GetPaddings>, ICostEvaluator<GetPaddings>, IShapeEvaluator<GetPaddings>, IMetricEvaluator<GetPaddings>
{
    public static Expr ConcatPadding(Expr[] padH, Expr[] padW)
    {
        // return [[padh_before, padh_after],
        //         [padw_before, padw_after]]
        return Stack(
            new IR.Tuple(
                Stack(new IR.Tuple(padH), 0),
                Stack(new IR.Tuple(padW), 0)),
            0);
    }


    public IValue Visit(IEvaluateContext context, GetPaddings target)
    {
        var inShape = context.GetArgumentValueAsArray<int>(target, GetPaddings.InputShape);
        var wShape = context.GetArgumentValueAsArray<int>(target, GetPaddings.WeightsShape);
        var strides = context.GetArgumentValueAsArray<int>(target, GetPaddings.Strides);
        var dilations = context.GetArgumentValueAsArray<int>(target, GetPaddings.Dilations);
        var same = context.GetArgumentValueAsScalar<bool>(target, GetPaddings.Same);
        var lower = context.GetArgumentValueAsScalar<bool>(target, GetPaddings.Lower);
        var padH = GetWindowedPadding(inShape[2], wShape[2], strides[0], dilations[0], same, lower);
        var padW = GetWindowedPadding(inShape[3], wShape[3], strides[1], dilations[1], same, lower);
        return ConcatPadding(padH, padW).Evaluate();
    }

    public static Expr[] GetWindowedPadding(Expr inputSize, Expr filter, Expr stride, Expr dilation, bool same, bool lower = false)
    {
        var i32InputSize = Cast(inputSize, DataTypes.Int32);
        var i32Filter = Cast(filter, DataTypes.Int32);
        var i32Stride = Cast(stride, DataTypes.Int32);
        var i32Dilation = Cast(dilation, DataTypes.Int32);
        var outputSize = IR.Util.GetWindowedOutputSize(i32InputSize, i32Filter, i32Stride, i32Dilation, same, false);
        return GetWindowedPaddingValue(i32InputSize, outputSize, i32Filter, i32Stride, i32Dilation, lower);
    }

    public IRType Visit(ITypeInferenceContext context, GetPaddings target)
    {
        return new TensorType(DataTypes.Int64, new[] { 2, 2 });
    }

    private static Expr[] GetWindowedPaddingValue(Expr inputSize, Expr outputSize, Expr filter, Expr stride, Expr dilation, bool lower)
    {
        var effectiveFilterSize = ((filter - 1) * dilation) + 1;
        var padding = IR.F.Math.Max(0, ((outputSize - 1) * stride) + effectiveFilterSize - inputSize);
        var before = Cast(padding / 2, DataTypes.Int32);
        var after = Cast(padding - (padding / 2), DataTypes.Int32);
        if (lower)
        {
            return new[] { IR.F.Math.Max(before, after), IR.F.Math.Min(before, after) };
        }

        return new[] { before, after };
    }

    public Cost Visit(ICostEvaluateContext context, GetPaddings target)
    {
        return CostUtility.GetShapeExprCost();
    }

    public Expr Visit(IShapeEvaluateContext context, GetPaddings target)
    {
        return new[] { 2, 2 };
    }

    public Metric Visit(IMetricEvaluateContext context, GetPaddings target)
    {
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType),
        };
    }
}
