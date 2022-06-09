// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Imaging;
using Nncase.IR.Tensors;
using OrtKISharp;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Imaging;

/// <summary>
/// Evaluator for <see cref="ResizeImage"/>.
/// </summary>
public class ResizeImageEvaluator : IEvaluator<ResizeImage>, ITypeInferencer<ResizeImage>
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
        input = tf.transpose(input, new[] { 0, 2, 3, 1 });
        var sizes = context.GetArgumentValueAsArray<int>(target, ResizeImage.NewSize);
        var halfPixelCenter = target.TransformationMode == ImageResizeTransformationMode.HalfPixel;
        var alignCorners = target.TransformationMode == ImageResizeTransformationMode.AlignCorners;
        var size = new NDArray(new[] { sizes[2], sizes[3] }, new[] { 2 });
        var output = target.ResizeMode switch
        {
            ImageResizeMode.Bilinear => tf.image.resize_bilinear(input, size, alignCorners, halfPixelCenter),
            ImageResizeMode.NearestNeighbor => tf.image.resize_nearest_neighbor(input, size, alignCorners, "",
                halfPixelCenter),
            _ => throw new NotSupportedException($"TFResize Not suppoprted {target.ResizeMode}")
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
        return OrtKI.ResizeWithSizes(input, roi, sizes,
            ResizeModeHelper.ToString(target.TransformationMode),
            cubicCoeffA, excludeOutside, extrapolationValue,
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
}
