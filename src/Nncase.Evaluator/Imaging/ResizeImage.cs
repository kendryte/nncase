// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Imaging;
using Nncase.IR.Tensors;
using OrtKISharp;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Evaluator.Imaging;

/// <summary>
/// Evaluator for <see cref="ResizeImage"/>.
/// </summary>
public class ResizeImageEvaluator : IEvaluator<ResizeImage>, ITypeInferencer<ResizeImage>, ICostEvaluator<ResizeImage>, IMetricEvaluator<ResizeImage>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, ResizeImage target)
    {
        return OnnxResize(context, target);
    }

    public IValue OnnxResize(IEvaluateContext context, ResizeImage target)
    {
        var input = context.GetOrtArgumentValue(target, ResizeImage.Input);
        var roi = context.GetOptionalOrtArgumentValue(target, ResizeImage.Roi, Array.Empty<float>());
        var sizes = context.GetInt64OrtTensorArgumentValue(target, ResizeImage.NewSize);
        var cubicCoeffA = context.GetOptionArgumentValueAsScalar<float>(target, ResizeImage.CubicCoeffA, target.IsTFResize ? -0.5f : -0.75f);
        var excludeOutside = context.GetOptionArgumentValueAsScalar<long>(target, ResizeImage.ExcludeOutside, 0);
        var extrapolationValue = context.GetOptionArgumentValueAsScalar<float>(target, ResizeImage.ExtrapolationValue, 0f);

        // when HasBindedMixQuantInfo is true, eval will do simulation of quant/dequant for some inputs, this is used for evaluate accumulated quant error for layers.
        if (context.CurrentCall.EnodeBestQuantConfigWithCosine != null)
        {
            var pattern = IsRangeOfMarker(IsWildcard(), IsWildcard());
            if (pattern.MatchLeaf(context.CurrentCall.Arguments.ToArray()[0]) && ((Nncase.IR.Marker)context.CurrentCall.Arguments.ToArray()[0]).MixQuantInfo?.HasBindedMixQuantInfo == true)
            {
                var quantParam = ((Nncase.IR.Marker)context.CurrentCall.Arguments.ToArray()[0]).MixQuantInfo!.QuantParameter;

                // input feature map quantParam count should be 1 since input feature map quant is by tensor.
                Trace.Assert(quantParam.Count == 1);
                var inputFloat = input.ToArray<float>();
                for (var i = 0; i < inputFloat.Length; i++)
                {
                    var inputBufQuant = (double)((inputFloat[i] / (double)quantParam[0].Scale) + quantParam[0].ZeroPoint);
                    if (!(quantParam[0].Scale == 1.0f && quantParam[0].ZeroPoint == 0))
                    {
                        inputBufQuant = System.Math.Round((double)(float)inputBufQuant);
                    }

                    var inputBufDeQuant = (float)((inputBufQuant - quantParam[0].ZeroPoint) * (double)quantParam[0].Scale);
                    inputFloat[i] = (float)inputBufDeQuant;
                }

                input = OrtKISharp.Tensor.MakeTensor(inputFloat, input.Shape);
            }
        }

        if (target.IsTFResize)
        {
            var transformationMode = "asymmetric";
            var nearestMode = "floor";
            if (target.TransformationMode == ImageResizeTransformationMode.AlignCorners)
            {
                transformationMode = "align_corners";
                nearestMode = "round_prefer_ceil";
            }
            else if (target.TransformationMode == ImageResizeTransformationMode.HalfPixel)
            {
                transformationMode = target.ResizeMode switch
                {
                    ImageResizeMode.NearestNeighbor => "tf_half_pixel_for_nn",
                    _ => "half_pixel",
                };
            }

            return OrtKI.ResizeWithSizes(
                input,
                roi.Cast(OrtDataType.Float),
                sizes,
                transformationMode,
                cubicCoeffA,
                excludeOutside,
                extrapolationValue,
                ResizeModeHelper.ToString(target.ResizeMode),
                nearestMode).ToValue();
        }
        else
        {
            return OrtKI.ResizeWithSizes(
                input,
                roi,
                sizes,
                ResizeModeHelper.ToString(target.TransformationMode),
                cubicCoeffA,
                excludeOutside,
                extrapolationValue,
                ResizeModeHelper.ToString(target.ResizeMode),
                ResizeModeHelper.ToString(target.NearestMode)).ToValue();
        }
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, ResizeImage target)
    {
        var input = context.CheckArgumentType<IRType>(target, ResizeImage.Input);
        var newSize = context.GetArgument(target, ResizeImage.NewSize);

        return input switch
        {
            TensorType t => Visit(t, newSize),
            DistributedType d => Visit(d, newSize),
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    public IRType Visit(TensorType input, Expr newSize)
    {
        return TypeInference.ResizeType(input, newSize, null);
    }

    public IRType Visit(DistributedType input, Expr newSize)
    {
        if (Visit(input.TensorType, newSize) is not TensorType tensorType)
        {
            return new InvalidType(string.Empty);
        }

        var ndsbp = new SBP[input.Placement.Rank];

        var invalid = new InvalidType($"{input}, not support");
        for (int i = 0; i < input.Placement.Rank; i++)
        {
            switch (input.NdSBP[i])
            {
                case SBPSplit { Axis: int ix } when ix < 2:
                    ndsbp[i] = SBP.S(ix);
                    break;
                case SBPBroadCast:
                    ndsbp[i] = SBP.B;
                    break;
                default:
                    return invalid;
            }
        }

        return new DistributedType(tensorType, ndsbp, input.Placement);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, ResizeImage target)
    {
        var inputType = context.GetArgumentType<IRType>(target, ResizeImage.Input);
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, ResizeImage target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, ResizeImage.Input);
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(returnType),
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(returnType) * MetricUtility.ResizeLinearFLOPs,
        };
    }
}
