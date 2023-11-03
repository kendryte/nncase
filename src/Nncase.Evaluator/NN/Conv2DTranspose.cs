﻿// Copyright (c) Canaan Inc. All rights reserved.
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
public class Conv2DTransposeEvaluator : IEvaluator<Conv2DTranspose>, ITypeInferencer<Conv2DTranspose>, ICostEvaluator<Conv2DTranspose>, IShapeEvaluator<Conv2DTranspose>, IMetricEvaluator<Conv2DTranspose>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Conv2DTranspose conv)
    {
        var input = context.GetOrtArgumentValue(conv, Conv2DTranspose.Input);
        var weights = context.GetOrtArgumentValue(conv, Conv2DTranspose.Weights);
        var bias = context.GetOrtArgumentValue(conv, Conv2DTranspose.Bias);
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

        var gIC = inputShape[1] / groups;
        var gOC = outputShape[1] / groups;

        var weightsArray = weights.ToArray<float>();
        var inputsArray = input.ToArray<float>();
        var biasArray = bias.ToArray<float>();
        int inputIndex = 0;
        for (int batch = 0; batch < inputShape[0]; batch++)
        {
            var outBatchP = outCache.AsSpan().Slice(batch * (int)outputShape[1] * (int)outputShape[2] * (int)outputShape[3]);

            for (int g = 0; g < groups; g++)
            {
                var outGroupP = outBatchP.Slice(g * (int)gOC * (int)outputShape[2] * (int)outputShape[3]);
                var wGroupP = weightsArray.AsSpan().Slice((int)g * (int)gOC * (int)gIC * (int)kernelShape[2] * (int)kernelShape[3]);

                for (int ic = 0; ic < gIC; ic++)
                {
                    for (int iy = 0; iy < inputShape[2]; iy++)
                    {
                        for (int ix = 0; ix < inputShape[3]; ix++)
                        {
                            int outYOrigin = (int)((iy * stride[0]) - pads[0]);
                            int outXOrigin = (int)((ix * stride[1]) - pads[2]);
                            int filterYStart = System.Math.Max(0, (int)((-outYOrigin + dilation[0] - 1) / dilation[0]));
                            int filterYEnd = (int)System.Math.Min(kernelShape[2], ((int)outputShape[2] - outYOrigin + dilation[0] - 1) / dilation[0]);
                            int filterXStart = (int)System.Math.Max(0, (-outXOrigin + dilation[1] - 1) / dilation[1]);
                            int filterXEnd = (int)System.Math.Min(kernelShape[3], ((int)outputShape[3] - outXOrigin + dilation[1] - 1) / dilation[1]);

                            float inV;
                            if (ix < 0 || ix >= inputShape[3] || iy < 0 || iy >= inputShape[2])
                            {
                                inV = 0f;
                            }
                            else
                            {
                                inV = inputsArray[inputIndex];
                            }

                            inputIndex++;

                            for (int oc = 0; oc < gOC; oc++)
                            {
                                var outCP = outGroupP.Slice((int)(oc * outputShape[2] * outputShape[3]));
                                var wOCP = wGroupP.Slice((int)(oc * gIC * kernelShape[2] * kernelShape[3]));
                                var wICP = wOCP.Slice((int)(ic * kernelShape[2] * kernelShape[3]));

                                for (int ky = filterYStart; ky < filterYEnd; ky++)
                                {
                                    for (int kx = filterXStart; kx < filterXEnd; kx++)
                                    {
                                        int outY = (int)(outYOrigin + (dilation[0] * ky));
                                        int outX = (int)(outXOrigin + (dilation[1] * kx));

                                        var w = wICP[(int)((ky * kernelShape[3]) + kx)];

                                        outCP[(int)((outY * outputShape[3]) + outX)] += (float)inV * w;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for (int i = 0; i < outputSize; i++)
        {
            var biasIdx = i / (outputShape[2] * outputShape[3]) % outputShape[1];
            outCache[i] = outCache[i] + biasArray[biasIdx];
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
        return new() { [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(weightsType) + CostUtility.GetMemoryAccess(biasType), [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType), [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, (uint)macPerElement.FixedValue * 2), };
    }

    public Metric Visit(IMetricEvaluateContext context, Conv2DTranspose target)
    {
        var returnType = context.GetReturnType<TensorType>();
        _ = returnType.Shape.ToValueArray();

        var inputType = context.GetArgumentType<TensorType>(target, Conv2DTranspose.Input);
        var inputShape = inputType.Shape.ToValueArray();
        var weightType = context.GetArgumentType<TensorType>(target, Conv2DTranspose.Weights);
        var weightShape = weightType.Shape.ToValueArray();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(weightType) + CostUtility.GetMemoryAccess(returnType),
            [MetricFactorNames.FLOPs] = (UInt128)(inputShape[0] * weightShape[0] * weightShape[1] * inputShape[2] * inputShape[3] * weightShape[2] * weightShape[3]),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Conv2DTranspose target)
    {
        var input = context.GetArgumentShape(target, Conv2DTranspose.Input);
        var weights = context.GetArgumentShape(target, Conv2DTranspose.Weights);
        var stride = context.GetArgument(target, Conv2DTranspose.Stride);
        var dilation = context.GetArgument(target, Conv2DTranspose.Dilation);
        var padding = context.GetArgument(target, Conv2DTranspose.Padding);
        var outputPadding = context.GetArgument(target, Conv2DTranspose.OutputPadding);
        var groups = context.GetArgument(target, Conv2DTranspose.Groups);
        return IR.F.ShapeExpr.Conv2DTransposeShape(IR.F.Tensors.Cast(input, DataTypes.Int64), weights, stride, dilation, padding, outputPadding, groups);
    }
}
