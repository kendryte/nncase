// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="Conv2DTranspose"/>.
/// </summary>
public class Conv2DTransposeEvaluator : IEvaluator<Conv2DTranspose>, ITypeInferencer<Conv2DTranspose>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Conv2DTranspose conv)
    {
        var input = context.GetOrtArgumentValue(conv, Conv2DTranspose.Input);
        var weights = context.GetOrtArgumentValue(conv, Conv2DTranspose.Weights);
        var bias = context.GetOrtArgumentValue(conv, Conv2DTranspose.Bias);
        var stride = context.GetArgumentValueAsArray<long>(conv, Conv2DTranspose.Stride);
        var outputShape = context.GetArgumentValueAsArray<long>(conv, Conv2DTranspose.OutputShape);
        // [w:[left right] h:[top bottom]]
        var pads = context.GetArgumentValueAsArray<long>(conv, Conv2DTranspose.Padding);
        var outputPaddings = context.GetArgumentValueAsArray<long>(conv, Conv2DTranspose.OutputPadding);
        var dilation = context.GetArgumentValueAsArray<long>(conv, Conv2DTranspose.Dilation);
        var groups = context.GetArgumentValueAsScalar<long>(conv, Conv2DTranspose.Groups);
        var kernelShape = weights.Shape;
        return OrtKI.ConvTranspose(input, weights, bias, "NOTSET", dilation, groups,
            new long[] {kernelShape[2], kernelShape[3]}, outputPaddings, 
            outputShape, pads, stride).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Conv2DTranspose target)
    {
        if (context.GetArgument(target, Conv2DTranspose.OutputShape) is TensorConst outShapeValue)
        {
            var input = context.CheckArgumentType<TensorType>(target, Conv2DTranspose.Input);
            return new TensorType(input.DType, new Shape(outShapeValue.Value.Cast<int>()));
        }
        else
        {
            return new InvalidType("Conv2dTranspose can't infer shape with dynamic outputShape");
        }
    }
}
