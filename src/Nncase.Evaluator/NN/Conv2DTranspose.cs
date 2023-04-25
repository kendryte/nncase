// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;
using static Nncase.Evaluator.EvaluatorUtil;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="Conv2DTranspose"/>.
/// </summary>
public class Conv2DTransposeEvaluator : IEvaluator<Conv2DTranspose>, ITypeInferencer<Conv2DTranspose>, ICostEvaluator<Conv2DTranspose>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Conv2DTranspose conv)
    {
        var input = context.GetOrtArgumentValue(conv, Conv2DTranspose.Input);
        var weights = context.GetOrtArgumentValue(conv, Conv2DTranspose.Weights);
        _ = context.GetOrtArgumentValue(conv, Conv2DTranspose.Bias);
        var stride = context.GetArgumentValueAsArray<long>(conv, Conv2DTranspose.Stride);
        var outputShape = context.GetArgumentValueAsArray<long>(conv, Conv2DTranspose.OutputShape);

        // [h:[top bottom] w:[left right] ]
        var pads = context.GetArgumentValueAsArray<long>(conv, Conv2DTranspose.Padding);
        _ = context.GetArgumentValueAsArray<long>(conv, Conv2DTranspose.OutputPadding);
        var dilation = context.GetArgumentValueAsArray<long>(conv, Conv2DTranspose.Dilation);
        var groups = context.GetArgumentValueAsScalar<long>(conv, Conv2DTranspose.Groups);
        var kernelShape = weights.Shape;
        var inputShape = input.Shape;

        var outputSize = outputShape[0] * outputShape[1] * outputShape[2] * outputShape[3];
        float[] outCache = new float[outputSize];
        Array.Clear(outCache, 0, (int)outputSize);

        var g_ic = inputShape[1] / groups;
        var g_oc = outputShape[1] / groups;

        int inputIndex = 0;
        for (int batch = 0; batch < inputShape[0]; batch++)
        {
            var out_batch_p = outCache.AsSpan().Slice(batch * (int)outputShape[1] * (int)outputShape[2] * (int)outputShape[3]);

            for (int g = 0; g < groups; g++)
            {
                var out_group_p = out_batch_p.Slice(g * (int)g_oc * (int)outputShape[2] * (int)outputShape[3]);
                var w_group_p = weights.ToArray<float>().AsSpan().Slice((int)g * (int)g_oc * (int)g_ic * (int)kernelShape[2] * (int)kernelShape[3]);

                for (int ic = 0; ic < g_ic; ic++)
                {
                    for (int iy = 0; iy < inputShape[2]; iy++)
                    {
                        for (int ix = 0; ix < inputShape[3]; ix++)
                        {
                            int out_y_origin = (int)((iy * stride[0]) - pads[0]);
                            int out_x_origin = (int)((ix * stride[1]) - pads[2]);
                            int filter_y_start = System.Math.Max(0, (int)((-out_y_origin + dilation[0] - 1) / dilation[0]));
                            int filter_y_end = (int)System.Math.Min(kernelShape[2], ((int)outputShape[2] - out_y_origin + dilation[0] - 1) / dilation[0]);
                            int filter_x_start = (int)System.Math.Max(0, (-out_x_origin + dilation[1] - 1) / dilation[1]);
                            int filter_x_end = (int)System.Math.Min(kernelShape[3], ((int)outputShape[3] - out_x_origin + dilation[1] - 1) / dilation[1]);

                            float in_v;
                            if (ix < 0 || ix >= inputShape[3] || iy < 0 || iy >= inputShape[2])
                            {
                                in_v = 0f;
                            }
                            else
                            {
                                in_v = input.ToArray<float>()[inputIndex];
                            }

                            inputIndex++;

                            for (int oc = 0; oc < g_oc; oc++)
                            {
                                var out_c_p = out_group_p.Slice((int)(oc * outputShape[2] * outputShape[3]));
                                var w_oc_p = w_group_p.Slice((int)(oc * g_ic * kernelShape[2] * kernelShape[3]));
                                var w_ic_p = w_oc_p.Slice((int)(ic * kernelShape[2] * kernelShape[3]));

                                for (int ky = filter_y_start; ky < filter_y_end; ky++)
                                {
                                    for (int kx = filter_x_start; kx < filter_x_end; kx++)
                                    {
                                        int out_y = (int)(out_y_origin + (dilation[0] * ky));
                                        int out_x = (int)(out_x_origin + (dilation[1] * kx));

                                        var w = w_ic_p[(int)((ky * kernelShape[3]) + kx)];

                                        out_c_p[(int)((out_y * outputShape[3]) + out_x)] += (float)in_v * w;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return new TensorValue(Tensor.From(outCache, new[] { (int)outputShape[0], (int)outputShape[1], (int)outputShape[2], (int)outputShape[3] }));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Conv2DTranspose target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Conv2DTranspose.Input);
        if (context.GetArgument(target, Conv2DTranspose.OutputShape) is TensorConst outShapeValue)
        {
            return new TensorType(input.DType, new Shape(outShapeValue.Value.Cast<int>()));
        }
        else
        {
            return input with { Shape = Shape.Unknown(4) };
        }
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Conv2DTranspose target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Conv2DTranspose.Input);
        var weightsType = context.GetArgumentType<TensorType>(target, Conv2DTranspose.Weights);
        var biasType = context.GetArgumentType<TensorType>(target, Conv2DTranspose.Bias);
        var weightsShape = context.GetArgumentType<TensorType>(target, Conv2DTranspose.Weights).Shape;
        var outputType = context.GetReturnType<TensorType>();

        var macPerElement = weightsShape[1] * weightsShape[2] * weightsShape[3];
        return new() { [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(weightsType) + CostUtility.GetMemoryAccess(biasType), [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType), [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, macPerElement.FixedValue * 2), };
    }
}
