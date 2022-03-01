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

namespace Nncase.Evaluator.Imaging;

/// <summary>
/// Evaluator for <see cref="ResizeImage"/>.
/// </summary>
public class ResizeImageEvaluator : IEvaluator<ResizeImage>, ITypeInferencer<ResizeImage>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, ResizeImage target)
    {
        var input = context.GetOrtArgumentValue(target, ResizeImage.Input);
        var roi = context.GetOrtArgumentValue(target, ResizeImage.Roi);
        var sizes = context.GetOrtArgumentValue(target, ResizeImage.NewSize);
        var cubicCoeffA = context.GetArgumentValueAsScalar<float>(target, ResizeImage.CubicCoeffA);
        var excludeOutside = context.GetArgumentValueAsScalar<long>(target, ResizeImage.ExcludeOutside);
        var extrapolationValue = context.GetArgumentValueAsScalar<float>(target, ResizeImage.ExtrapolationValue);
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
        return TypeInference.ResizeType(input, newSize);
    }
}
