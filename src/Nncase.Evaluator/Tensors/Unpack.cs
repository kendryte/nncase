// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
#pragma warning disable SA1010, SA1008
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using DryIoc.ImTools;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

public sealed class UnpackEvaluator : ITypeInferencer<Unpack>, ICostEvaluator<Unpack>, IEvaluator<Unpack>
{
    public static Tensor UnpackImpl(Tensor input, IRArray<int> axes)
    {
        if (axes.Count == 0)
        {
            return input;
        }

        var (oldLanesCount, basicElemType) = input.ElementType switch
        {
            VectorType vt2 => (vt2.Lanes.Count, vt2.ElemType),
            MaskVectorType => (1, DataTypes.Boolean),
            _ => throw new InvalidOperationException($"Unsupported input type: {input.ElementType}"),
        };

        var remainLanes = oldLanesCount - axes.Count;

        var preType = input.ElementType.Legalize((DataTypes.Float8E4M3, DataTypes.UInt8), (DataTypes.Float8E5M2, DataTypes.UInt8));
        var postType = remainLanes switch
        {
            0 => basicElemType,
            > 0 when input.ElementType is VectorType vt => new VectorType(basicElemType, vt.Lanes.Skip(remainLanes).ToArray()),
            _ => throw new InvalidOperationException($"Unsupported remain lanes: {remainLanes}"),
        };

        return input.ToOrtTensor(preType).Unpack(oldLanesCount, axes).ToTensor(postType);
    }

    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Unpack target)
    {
        var input = context.GetArgumentValueAsTensor(target, Unpack.Input);
        return Value.FromTensor(UnpackImpl(input, target.Axes));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Unpack target)
    {
        var input = context.CheckArgumentType<IRType>(target, Unpack.Input);

        return input switch
        {
            DistributedType d => Visit(context, target, d),
            TensorType t => Visit(context, target, t),
            AnyType => AnyType.Default,
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Unpack target)
    {
        var inputType = context.GetArgumentType<IRType>(target, Unpack.Input);
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Unpack target)
    {
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
        };
    }

    private IRType Visit(ITypeInferenceContext context, Unpack target, TensorType input)
    {
        if (target.Lanes.Any(x => x <= 0))
        {
            return new InvalidType("devectorize lane <= 0");
        }

        return TypeInference.UnpackType(input, target.Axes);
    }

    private IRType Visit(ITypeInferenceContext context, Unpack target, DistributedType input)
    {
        if (Visit(context, target, input.TensorType) is not TensorType tensorType)
        {
            throw new InvalidOperationException();
        }

        // [m]<8>@8@4 -> [m*8]@8@4, when max(m)=256 and runtime m=12, input and output have different local shape.
        foreach (var (s, r) in input.AxisPolicies.Select((s, r) => (s, r)))
        {
            if (s is SBPSplit split && target.Axes.Contains(r))
            {
                var divisor = split.Axes.Select(a => input.Placement.Hierarchy[a]).Aggregate(1, (a, b) => a * b);
                var dim = input.TensorType.Shape[r];
                if (!(dim.IsFixed && dim.FixedValue % divisor == 0))
                {
                    return new InvalidType("Not support non-divisible input");
                }
            }
        }

        return new DistributedType(tensorType, input.AxisPolicies, input.Placement);
    }
}
