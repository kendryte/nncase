// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Imaging;
using Nncase.IR.Tensors;
using OrtKISharp;
using Tensorflow;
using Tensorflow.NumPy;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Imaging;

/// <summary>
/// Evaluator for <see cref="ResizeImage"/>.
/// </summary>
public class ResizeImageEvaluator : IEvaluator<ResizeImage>, ITypeInferencer<ResizeImage>, ICostEvaluator<ResizeImage>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, ResizeImage target)
    {
        return target.IsTFResize
            ? TFResize(context, target)
            : OnnxResize(context, target);
    }

    public IValue TFResize(IEvaluateContext context, ResizeImage target)
    {
        var input = context.GetTFArgumentValue(target, ResizeImage.Input);

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

                input = tf.constant(inputFloat, TF_DataType.TF_FLOAT, input.shape);
            }
        }

        input = tf.transpose(input, new[] { 0, 2, 3, 1 });
        var sizes = context.GetArgumentValueAsArray<int>(target, ResizeImage.NewSize);
        var halfPixelCenter = target.TransformationMode == ImageResizeTransformationMode.HalfPixel;
        var alignCorners = target.TransformationMode == ImageResizeTransformationMode.AlignCorners;
        var size = new NDArray(new[] { sizes[2], sizes[3] }, new[] { 2 });
        var output = target.ResizeMode switch
        {
            ImageResizeMode.Bilinear => tf.image.resize_bilinear(input, size, alignCorners, halfPixelCenter),
            ImageResizeMode.NearestNeighbor => tf.image.resize_nearest_neighbor(input, size, alignCorners, string.Empty, halfPixelCenter),
            _ => throw new NotSupportedException($"TFResize Not suppoprted {target.ResizeMode}"),
        };
        return tf.transpose(output, new[] { 0, 3, 1, 2 }).ToValue();
    }

    public IValue OnnxResize(IEvaluateContext context, ResizeImage target)
    {
        var input = context.GetOrtArgumentValue(target, ResizeImage.Input);
        var roi = context.GetOptionalOrtArgumentValue(target, ResizeImage.Roi, Array.Empty<float>());
        var sizes = context.GetInt64OrtTensorArgumentValue(target, ResizeImage.NewSize);
        var cubicCoeffA = context.GetOptionArgumentValueAsScalar<float>(target, ResizeImage.CubicCoeffA, -0.75f);
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

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, ResizeImage target)
    {
        var input = context.CheckArgumentType<TensorType>(target, ResizeImage.Input);
        var newSize = context.GetArgument(target, ResizeImage.NewSize);
        return TypeInference.ResizeType(input, newSize, null);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, ResizeImage target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, ResizeImage.Input);
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }
}
