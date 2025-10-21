// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Runtime.CompilerServices;
using DryIoc.ImTools;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.TIR.NTT;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.IR.F.Tensors;
using LayerNorm = Nncase.IR.CustomNTT.LayerNorm;

namespace Nncase.Evaluator.CustomNTT;

/// <summary>
/// Evaluator for <see cref="LayerNorm"/>.
/// </summary>
public class LayerNormEvaluator : IEvaluator<LayerNorm>, ITypeInferencer<LayerNorm>, ICostEvaluator<LayerNorm>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, LayerNorm layerNorm)
    {
        var dataType = context.CurrentCall.Arguments[LayerNorm.Input.Index].CheckedDataType;
        var input = context.GetArgumentValueAsTensor<float>(layerNorm, LayerNorm.Input);
        var scale = context.GetArgumentValueAsTensor<float>(layerNorm, LayerNorm.Scale);
        var bias = context.GetArgumentValueAsTensor<float>(layerNorm, LayerNorm.Bias);
        var postScale = context.GetArgumentValue(layerNorm, LayerNorm.Scale).AsTensor();
        var ret = Tensor.From(NN.LayerNormEvaluator.LayerNormImpl(input.Shape.ToValueArray(), input.Buffer.Span, scale.Buffer.Span, bias.Buffer.Span, layerNorm.Axis, layerNorm.Epsilon, layerNorm.UseMean), input.Shape.ToValueArray()).CastTo(dataType);

        return OrtKI.Cast(ret.ToOrtTensor() * postScale.ToOrtTensor(), (long)layerNorm.OutputDataType.ToOrtType()).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, LayerNorm target)
    {
        var input = context.CheckArgumentType<IRType>(target, LayerNorm.Input);
        var scale = context.CheckArgumentType<IRType>(target, LayerNorm.Scale);
        var bias = context.CheckArgumentType<IRType>(target, LayerNorm.Bias);

        if (CheckCustomSBP(input, scale, bias, target))
        {
            return (input, scale, bias) switch
            {
                (DistributedType a, DistributedType b, DistributedType c) => new DistributedType((TensorType)VisitTensorType(target, a.TensorType, b.TensorType, c.TensorType), target.OutSBPs, a.Placement),
                (TensorType a, TensorType b, TensorType c) => VisitTensorType(target, a, b, c),
                _ => new InvalidType($"{input} {scale} {bias} not support"),
            };
        }
        else
        {
            return new InvalidType("Not Match With CustomSBP!");
        }
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, LayerNorm target)
    {
        return target.Cost;
    }

    private bool CheckCustomSBP(IRType input, IRType scale, IRType bias, LayerNorm layerNorm)
    {
        if (input is DistributedType a && scale is DistributedType b && bias is DistributedType c)
        {
            if (Enumerable.Range(0, a.TensorType.Shape.Rank).Any(i => a.AxisPolicies[i] != layerNorm.InSBPs[i]))
            {
                return false;
            }

            if (Enumerable.Range(0, b.TensorType.Shape.Rank).Any(i => b.AxisPolicies[i] != layerNorm.ScaleSBPs[i]))
            {
                return false;
            }

            if (Enumerable.Range(0, c.TensorType.Shape.Rank).Any(i => c.AxisPolicies[i] != layerNorm.BiasSBPs[i]))
            {
                return false;
            }
        }

        return true;
    }

    private IRType VisitTensorType(LayerNorm target, TensorType input, TensorType scale, TensorType bias)
    {
        if (input.Shape.IsUnranked || scale.Shape.IsUnranked || bias.Shape.IsUnranked)
        {
            return new TensorType(target.OutputDataType, Shape.Unranked);
        }

        if (input.DType is VectorType vt)
        {
            if (vt.Lanes.Count == 1)
            {
                var scaleV = 1f * target.OutputDataType.SizeInBytes / vt.ElemType.SizeInBytes;
                var newDType = new VectorType(target.OutputDataType, (int)(vt.Lanes[0] / scaleV));
                var newShape = input.Shape.ToArray();
                newShape[target.VectorizedAxes[0]] = scaleV > 1 ? newShape[^2] * (long)scaleV : newShape[^2] / (long)(1f / scaleV);
                return new TensorType(newDType, newShape);
            }
            else
            {
                return new InvalidType("Not supported vectorize.");
            }
        }
        else
        {
            return new TensorType(target.OutputDataType, input.Shape);
        }
    }
}
