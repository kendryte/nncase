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
/// Evaluator for <see cref="Conv2D"/>.
/// </summary>
public class Conv2DEvaluator : IEvaluator<Conv2D>, ITypeInferencer<Conv2D>, ICostEvaluator<Conv2D>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Conv2D conv)
    {
        var input = context.GetOrtArgumentValue(conv, Conv2D.Input);
        var weights = context.GetOrtArgumentValue(conv, Conv2D.Weights);
        var bias = context.GetOrtArgumentValue(conv, Conv2D.Bias);

        var stride = context.GetArgumentValueAsArray<long>(conv, Conv2D.Stride);
        var pad = context.GetInt64OrtTensorArgumentValue(conv, Conv2D.Padding);
        var dilation = context.GetArgumentValueAsArray<long>(conv, Conv2D.Dilation);
        var groups = context.GetArgumentValueAsScalar<long>(conv, Conv2D.Groups);
        var fusedClamp = context.GetArgumentValueAsArray<float>(conv, Conv2D.FusedClamp);
        var kernelShape = weights.Shape;
        var result = OrtKI.Conv(
            input,
            weights,
            bias,
            "NOTSET",
            dilation,
            groups,
            new long[] { kernelShape[2], kernelShape[3] },
            ToOnnxPadFormat(pad),
            stride);
        return OrtKI.Clip(result, fusedClamp[0], fusedClamp[1]).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Conv2D target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Conv2D.Input);
        var weights = context.CheckArgumentType<TensorType>(target, Conv2D.Weights);
        return Visit(context, target, input, weights);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Conv2D target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Conv2D.Input);
        var weightsType = context.GetArgumentType<TensorType>(target, Conv2D.Weights);
        var biasType = context.GetArgumentType<TensorType>(target, Conv2D.Bias);
        var outputType = context.GetReturnType<TensorType>();

        var weightsShape = weightsType.Shape;
        var macPerElement = (2 * weightsShape[1] * weightsShape[2] * weightsShape[3]) - 1;
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(weightsType) + CostUtility.GetMemoryAccess(biasType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, macPerElement.FixedValue),
        };
    }

    private IRType Visit(ITypeInferenceContext context, Conv2D target, TensorType input, TensorType weights)
    {
        var args = context.GetArguments(target, Conv2D.Stride, Conv2D.Padding, Conv2D.Dilation, Conv2D.Groups);
        return TypeInference.Conv2DType(input, weights, args[0], args[1], args[2], args[3]);
    }
}
