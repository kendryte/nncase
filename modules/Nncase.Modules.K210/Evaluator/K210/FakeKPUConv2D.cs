// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K210;
using OrtKISharp;
using static Nncase.Evaluator.K210EvaluatorUtil;

namespace Nncase.Evaluator.K210;

/// <summary>
/// Evaluator for <see cref="FakeKPUConv2D"/>.
/// </summary>
public class FakeKPUConv2DEvaluator : IEvaluator<FakeKPUConv2D>, ITypeInferencer<FakeKPUConv2D>, ICostEvaluator<FakeKPUConv2D>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, FakeKPUConv2D conv)
    {
        var input = context.GetOrtArgumentValue(conv, FakeKPUConv2D.Input);
        var weights = context.GetOrtArgumentValue(conv, FakeKPUConv2D.Weights);
        var bias = conv.Bias.ToOrtTensor();

        var stride = new long[] { 1, 1 };
        var pad = Enumerable.Repeat((long)KPUUtility.GetKPUPadding(conv.FilterType), 4).ToArray();
        var dilation = new long[] { 1, 1 };
        var groups = 1L;
        var kernelShape = weights.Shape;
        return OrtKI.Conv(input, weights, bias, "NOTSET", dilation, groups, new long[] { kernelShape[2], kernelShape[3] }, pad, stride).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, FakeKPUConv2D target)
    {
        var input = context.CheckArgumentType<TensorType>(target, FakeKPUConv2D.Input);
        var weights = context.CheckArgumentType<TensorType>(target, FakeKPUConv2D.Weights);
        return Visit(context, target, input, weights);
    }

    /// <inheritdoc/>
    public Cost? Visit(ICostEvaluateContext context, FakeKPUConv2D target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, FakeKPUConv2D.Input);
        var weightsType = context.GetArgumentType<TensorType>(target, FakeKPUConv2D.Weights);
        var weightsShape = context.GetArgumentType<TensorType>(target, FakeKPUConv2D.Weights).Shape;
        var outputType = context.GetReturnType<TensorType>();

        if (weightsShape.IsFixed)
        {
            var macPerElement = weightsShape[1] * weightsShape[2] * weightsShape[3];
            var kpuMac = target.FilterType == KPUFilterType.Filter_1x1 ? 64 : 64 * 9;
            return new()
            {
                [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(weightsType),
                [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
                [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, macPerElement.FixedValue * 2) / kpuMac,
            };
        }

        return null;
    }

    private IRType Visit(ITypeInferenceContext context, FakeKPUConv2D target, TensorType input, TensorType weights)
    {
        var stride = new[] { 1, 1 };
        var pad = Tensor.FromScalar(KPUUtility.GetKPUPadding(target.FilterType), new[] { 2, 2 });
        var dilation = new[] { 1, 1 };
        var groups = 1;
        return TypeInference.Conv2DType(input, weights, stride, pad, dilation, groups);
    }
}
