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

public sealed class PackEvaluator : ITypeInferencer<Pack>, ICostEvaluator<Pack>, IEvaluator<Pack>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Pack target)
    {
        if (context.CurrentCall.Arguments[Pack.Input.Index].CheckedDataType == DataTypes.Float8E4M3 || context.CurrentCall.Arguments[Pack.Input.Index].CheckedDataType == DataTypes.Float8E5M2)
        {
            var input = context.GetArgumentValueAsTensor(target, Pack.Input);
            var inputCasted = input.CastElement<float>();
            var inputOrt = inputCasted.ToOrtTensor();
            inputOrt = inputOrt.Pack(0, target.Lanes, target.Axes);
            var output = inputOrt.ToTensor().CastElementTo(context.CurrentCall.Arguments[Pack.Input.Index].CheckedDataType);
            output = output.CastTo(TypeInference.PackType(input.ElementType, target.Lanes), CastMode.Reinterpret);
            output = output.Squeeze(output.Rank - 1);
            return Value.FromTensor(output);
        }
        else
        {
            var input = context.GetArgumentValue(target, Pack.Input).AsTensor();
            var inputOrt = input.ToOrtTensor();
            inputOrt = inputOrt.Pack(0, target.Lanes, target.Axes);
            return inputOrt.ToValue(TypeInference.PackType(input.ElementType, target.Lanes));
        }
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
            if (input.AxisPolicies[i] is SBPSplit && target.Axes.Contains(i))
            {
                var lane = target.Lanes[target.Axes.IndexOf(i)];
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
