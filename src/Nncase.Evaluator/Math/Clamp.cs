// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Clamp"/>.
/// </summary>
public class ClampEvaluator : IEvaluator<Clamp>, ITypeInferencer<Clamp>, ICostEvaluator<Clamp>, IShapeEvaluator<Clamp>, IMetricEvaluator<Clamp>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Clamp clamp)
    {
        var input = context.GetOrtArgumentValue(clamp, Clamp.Input);
        var min = context.GetOrtArgumentValue(clamp, Clamp.Min);
        var max = context.GetOrtArgumentValue(clamp, Clamp.Max);
        return OrtKI.Min(new[] { OrtKI.Max(new[] { input, min }), max }).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Clamp target)
    {
        var input = context.CheckArgumentType<IRType>(target, Clamp.Input);
        var min = context.CheckArgumentType<TensorType>(target, Clamp.Min);
        var max = context.CheckArgumentType<TensorType>(target, Clamp.Max);

        return input switch
        {
            TensorType t => Visit(t, min, max),
            DistributedType d => Visit(d, min, max),
            _ => new InvalidType("Wrong Clamp Type!"),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Clamp target)
    {
        var inputType = context.GetArgumentType<IRType>(target, Clamp.Input);
        var minType = context.GetArgumentType<TensorType>(target, Clamp.Min);
        var maxType = context.GetArgumentType<TensorType>(target, Clamp.Max);
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(minType) + CostUtility.GetMemoryAccess(maxType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, 2),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Clamp target) => context.GetArgumentShape(target, Clamp.Input);

    public Metric Visit(IMetricEvaluateContext context, Clamp target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(outputType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType, 2),
        };
    }

    private IRType Visit(TensorType input, TensorType min, TensorType max)
    {
        if (input.DType != min.DType || input.DType != max.DType || min.DType != max.DType)
        {
            return new InvalidType(
                $"clamp type is not equal, input:{input.DType}, min:${min.DType}, max:${max.DType}");
        }

        if (TypeInference.BroadcastType(input, min) is InvalidType invalidMin)
        {
            return invalidMin;
        }

        if (TypeInference.BroadcastType(input, max) is InvalidType invalidMax)
        {
            return invalidMax;
        }

        if (min.Shape != max.Shape)
        {
            return new InvalidType($"The min.Shape {min.Shape} != max.Shape {max.Shape}");
        }

        return input;
    }

    private IRType Visit(DistributedType input, TensorType min, TensorType max)
    {
        return input;
    }
}
