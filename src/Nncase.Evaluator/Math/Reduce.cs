﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Diagnostics;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using Reduce = Nncase.IR.Math.Reduce;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Reduce"/>.
/// </summary>
public class ReduceEvaluator : IEvaluator<Reduce>, ITypeInferencer<Reduce>, ICostEvaluator<Reduce>, IMetricEvaluator<Reduce>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Reduce reduce)
    {
        var input = context.GetOrtArgumentValue(reduce, Reduce.Input);
        var axis = context.GetArgumentValueAsArray<long>(reduce, Reduce.Axes);
        var keepDims = context.GetArgumentValueAsScalar<long>(reduce, Reduce.KeepDims);

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

        return (reduce.ReduceOp switch
        {
            ReduceOp.Mean => OrtKI.ReduceMean(input, axis, keepDims),
            ReduceOp.Max => OrtKI.ReduceMax(input, axis, keepDims),
            ReduceOp.Min => OrtKI.ReduceMin(input, axis, keepDims),
            ReduceOp.Prod => OrtKI.ReduceProd(input, axis, keepDims),
            ReduceOp.Sum => OrtKI.ReduceSum(
                input,
                context.GetInt64OrtTensorArgumentValue(reduce, Reduce.Axes),
                keepDims,
                0),
            _ => throw new ArgumentOutOfRangeException(nameof(reduce)),
        }).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Reduce target)
    {
        var input = context.CheckArgumentType<IRType>(target, Reduce.Input);

        return input switch
        {
            DistributedType d => Visit(context, target, d),
            TensorType t => Visit(context, target, t),
            AnyType a => a,
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Reduce target)
    {
        var input = context.GetArgumentType<IRType>(target, Reduce.Input);
        var ret = context.GetReturnType<IRType>();
        var inputShape = input switch
        {
            TensorType t => t.Shape,
            DistributedType d => d.TensorType.Shape,
            _ => throw new NotSupportedException(string.Empty),
        };
        var retShape = ret switch
        {
            TensorType t => t.Shape,
            DistributedType d => d.TensorType.Shape,
            _ => throw new NotSupportedException(string.Empty),
        };
        uint input_elem = inputShape.Aggregate(1U, (acc, d) => acc * (d.IsFixed ? (uint)d.FixedValue : 1U));
        uint ret_elem = retShape.Aggregate(1U, (acc, d) => acc * (d.IsFixed ? (uint)d.FixedValue : 1U));
        uint macPerElement = input_elem / ret_elem;
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(ret, macPerElement),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Reduce target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Reduce.Input);
        var returnType = context.GetReturnType<TensorType>();
        var rF = MetricUtility.GetFLOPs(returnType);
        var iF = MetricUtility.GetFLOPs(inputType);
        var inner = iF / rF;
        _ = iF / inner;
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(returnType),
            [MetricFactorNames.FLOPs] = iF,
        };
    }

    private IRType Visit(ITypeInferenceContext context, Reduce target, TensorType input)
    {
        context.CheckArgumentType<TensorType>(target, Reduce.Axes);
        var args = context.GetArguments(target, Reduce.KeepDims, Reduce.Axes);
        return TypeInference.ReduceType(input, args[0], args[1]);
    }

    private IRType Visit(ITypeInferenceContext context, Reduce target, DistributedType input)
    {
        if (Visit(context, target, input.TensorType) is not TensorType tensorType)
        {
            throw new InvalidOperationException();
        }

        var axes = ((TensorConst)context.GetArgument(target, Reduce.Axes)).Value.ToArray<int>();

        var ndsbp = new SBP[input.TensorType.Shape.Rank];

        for (int i = 0; i < ndsbp.Length; i++)
        {
            switch (input.AxisPolices[i])
            {
                case SBPSplit split when axes.Contains(i):
                    // ndsbp[i] = SBP.P(target.ReduceOp);
                    return new InvalidType("reduce not support split on axes for now.");
                default:
                    ndsbp[i] = input.AxisPolices[i];
                    break;
            }
        }

        return new DistributedType(tensorType, tensorType.Shape.Rank == ndsbp.Length ? ndsbp : ndsbp.Where((_, i) => !axes.Contains(i)).ToArray(), input.Placement);
    }
}
