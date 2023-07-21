// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;
using static Nncase.Evaluator.EvaluatorUtil;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="ReduceWindow2D"/>.
/// </summary>
public class ReduceWindow2DEvaluator : IEvaluator<ReduceWindow2D>, ITypeInferencer<ReduceWindow2D>, ICostEvaluator<ReduceWindow2D>, IMetricEvaluator<ReduceWindow2D>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, ReduceWindow2D r)
    {
        var input = context.GetOrtArgumentValue(r, ReduceWindow2D.Input);
        var kernelSize = context.GetArgumentValueAsArray<long>(r, ReduceWindow2D.Filter);
        var stride = context.GetArgumentValueAsArray<long>(r, ReduceWindow2D.Stride);
        var dilation = context.GetArgumentValueAsArray<long>(r, ReduceWindow2D.Dilation);
        var pads = context.GetInt64OrtTensorArgumentValue(r, ReduceWindow2D.Padding);
        var countIncludePad = context.GetArgumentValueAsScalar<long>(r, ReduceWindow2D.CountIncludePad);
        var ceilMode = context.GetArgumentValueAsScalar<long>(r, ReduceWindow2D.CeilMode);
        var onnxPads = ToOnnxPadFormat(pads);

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

        return (r.ReduceOp switch
        {
            ReduceOp.Mean => OrtKI.AveragePool(input, "NOTSET", ceilMode, countIncludePad, kernelSize, onnxPads, stride),
            ReduceOp.Max => OrtKI.MaxPool(input, "NOTSET", ceilMode, dilation, kernelSize, onnxPads, countIncludePad, stride)[0],
            _ => throw new ArgumentOutOfRangeException(nameof(r)),
        }).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, ReduceWindow2D target)
    {
        var input = context.CheckArgumentType<TensorType>(target, ReduceWindow2D.Input);
        return Visit(context, target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, ReduceWindow2D target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, ReduceWindow2D.Input);
        var outputType = context.GetReturnType<TensorType>();

        uint macPerElement = 1;
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, macPerElement * 2),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, ReduceWindow2D target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, ReduceWindow2D.Input);
        var outputType = context.GetReturnType<TensorType>();

        var filter = context.GetArgument<TensorConst>(target, ReduceWindow2D.Filter).Value.ToArray<int>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(outputType),
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType, filter[0] * filter[1]),
        };
    }

    private IRType Visit(ITypeInferenceContext context, ReduceWindow2D target, TensorType input)
    {
        var args = context.GetArguments(target, ReduceWindow2D.Filter, ReduceWindow2D.Stride, ReduceWindow2D.Padding, ReduceWindow2D.CeilMode);
        return TypeInference.ReduceWindow2DType(input, args[0], args[1], args[2], args[3]);
    }
}
