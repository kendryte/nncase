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
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Unpack target)
    {
        var dt = context.CurrentCall.Arguments[Unpack.Input.Index].CheckedDataType;
        var elementType = dt is VectorType vt ? vt.ElemType : dt;
        if (elementType == DataTypes.Float8E4M3 || elementType == DataTypes.Float8E5M2)
        {
            var newType = new VectorType(DataTypes.Float32, target.Lanes.Select(l => l / 4).ToArray());
            var input = IR.F.Tensors.Cast(context.GetArgumentValue(target, Unpack.Input).AsTensor(), newType, CastMode.KDefault, target.Axes);
            var inputOrt = input.Evaluate().AsTensor().ToOrtTensor();

            foreach (var axis in target.Axes.Reverse())
            {
                inputOrt = inputOrt.Unpack(axis);
            }

            var output = IR.F.Tensors.Cast(inputOrt.ToTensor(), elementType).Evaluate().AsTensor();
            return Value.FromTensor(output);
        }
        else
        {
            var input = context.GetOrtArgumentValue(target, Unpack.Input);
            foreach (var axis in target.Axes.Reverse())
            {
                input = input.Unpack(axis);
            }

            return Value.FromTensor(input.ToTensor());
        }
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
            return new InvalidType("unpack lane <= 0");
        }

        return TypeInference.UnpackType(input, target.Axes);
    }

    private IRType Visit(ITypeInferenceContext context, Unpack target, DistributedType input)
    {
        if (Visit(context, target, input.TensorType) is not TensorType tensorType)
        {
            throw new InvalidOperationException();
        }

        // TODO: may support non-divisible input and pass it to output even if it's divisible.
        var shape = CompilerServices.GetMaxShape(input.TensorType.Shape);
        foreach (var (s, r) in input.AxisPolicies.Select((s, r) => (s, r)))
        {
            if (s is SBPSplit split)
            {
                var divisor = split.Axes.Select(a => input.Placement.Hierarchy[a]).Aggregate(1, (a, b) => a * b);
                if (shape[r] % divisor != 0)
                {
                    return new InvalidType("Not support non-divisible input");
                }
            }
        }

        return new DistributedType(tensorType, input.AxisPolicies, input.Placement);
    }
}
