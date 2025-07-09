// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
#pragma warning disable SA1010, SA1008
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Http.Headers;
using System.Numerics;
using System.Runtime.InteropServices;
using CommunityToolkit.HighPerformance;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

public sealed class PackMaskEvaluator : ITypeInferencer<PackMask>, ICostEvaluator<PackMask>, IEvaluator<PackMask>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, PackMask target)
    {
        var input = context.GetOrtArgumentValue(target, PackMask.Input);
        input = input.Pack(target.Lanes, target.Axis);
        return Value.FromTensor(input.ToTensor(new TensorType(new MaskVectorType(target.Style, target.ElementBits, target.Lanes), new RankedShape(input.Shape.SkipLast(1)))));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, PackMask target)
    {
        var input = context.CheckArgumentType<IRType>(target, PackMask.Input);

        return input switch
        {
            DistributedType d => Visit(context, target, d),
            TensorType t => Visit(context, target, t),
            AnyType => AnyType.Default,
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, PackMask target)
    {
        var inputType = context.GetArgumentType<IRType>(target, PackMask.Input);
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, PackMask target)
    {
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
        };
    }

    private IRType Visit(ITypeInferenceContext context, PackMask target, TensorType input)
    {
        return TypeInference.PackMaskType(input, target.Style, target.ElementBits, target.Lanes, target.Axis);
    }

    private IRType Visit(ITypeInferenceContext context, PackMask target, DistributedType input)
    {
        if (Visit(context, target, input.TensorType) is not TensorType tensorType)
        {
            throw new InvalidOperationException();
        }

        var divisor = Enumerable.Repeat(1, input.TensorType.Shape.Rank).ToList();
        for (int i = 0; i < divisor.Count; i++)
        {
            if (input.AxisPolicies[i] is SBPSplit split)
            {
                divisor[i] *= split.Axes.Select(s => input.Placement.Hierarchy[s]).Aggregate(1, (a, b) => a * b);
            }
        }

        var ndsbp = new SBP[input.TensorType.Shape.Rank];
        for (int i = 0; i < input.TensorType.Shape.Rank; i++)
        {
            if (input.AxisPolicies[i] is SBPSplit && target.Axis == i)
            {
                var lane = target.Lanes;
                if (input.TensorType.Shape[i] is { IsFixed: true, FixedValue: long s } && s / lane % divisor[i] == 0)
                {
                    ndsbp[i] = input.AxisPolicies[i];
                }
                else
                {
                    return new InvalidType($"{input}, not support");
                }
            }
            else
            {
                ndsbp[i] = input.AxisPolicies[i];
            }
        }

        return new DistributedType(tensorType, ndsbp, input.Placement);
    }
}
