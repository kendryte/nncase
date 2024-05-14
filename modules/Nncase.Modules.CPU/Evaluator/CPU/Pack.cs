// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
#pragma warning disable SA1010, SA1008
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.CPU;

public sealed class PackEvaluator : ITypeInferencer<Pack>, ICostEvaluator<Pack>, IEvaluator<Pack>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Pack target)
    {
        var input = context.GetOrtArgumentValue(target, Pack.Input);
        foreach (var (lanes, axis) in target.Lanes.Zip(target.Axes))
        {
            input = input.Pack(lanes, axis);
        }

        return Value.FromTensor(Tensor.FromBytes(new VectorType(input.DataType.ToDataType(), target.Lanes), input.BytesBuffer.ToArray(), input.Shape.ToArray().SkipLast(target.Lanes.Count).Select(i => (int)i).ToArray()));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Pack target)
    {
        var input = context.CheckArgumentType<IRType>(target, Pack.Input);

        return input switch
        {
            DistributedType d => Visit(context, target, d),
            TensorType t => Visit(context, target, t),
            AnyType => AnyType.Default,
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Pack target)
    {
        var inputType = context.GetArgumentType<IRType>(target, Pack.Input);
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Pack target)
    {
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
        };
    }

    private IRType Visit(ITypeInferenceContext context, Pack target, TensorType input)
    {
        return TypeInference.PackType(input, target.Lanes, target.Axes);
    }

    private IRType Visit(ITypeInferenceContext context, Pack target, DistributedType input)
    {
        if (Visit(context, target, input.TensorType) is not TensorType)
        {
            throw new InvalidOperationException();
        }

        var divisor = Enumerable.Repeat(1, input.TensorType.Shape.Rank).ToList();
        for (int i = 0; i < input.Placement.Rank; i++)
        {
            if (input.NdSBP[i] is SBPSplit { Axis: int axis })
            {
                divisor[axis] *= input.Placement.Hierarchy[i];
            }
        }

        var ndsbp = new SBP[input.Placement.Rank];
        for (int i = 0; i < input.Placement.Rank; i++)
        {
            if (input.NdSBP[i] is SBPSplit { Axis: int axis } && target.Axes.Contains(axis))
            {
                var lane = target.Lanes[target.Axes.IndexOf(axis)];
                if (input.TensorType.Shape[axis].FixedValue / lane % divisor[axis] == 0)
                {
                    ndsbp[i] = input.NdSBP[i];
                }
                else
                {
                    return new InvalidType($"{input}, not support");
                }
            }
            else
            {
                ndsbp[i] = input.NdSBP[i];
            }
        }

        return new DistributedType(input.TensorType, ndsbp, input.Placement);
    }
}
