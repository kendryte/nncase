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

public sealed class DevectorizeEvaluator : ITypeInferencer<Devectorize>, ICostEvaluator<Devectorize>, IEvaluator<Devectorize>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Devectorize target)
    {
        var dt = context.CurrentCall.Arguments[Devectorize.Input.Index].CheckedDataType;
        var elementType = dt is VectorType vt ? vt.ElemType : dt;
        if (elementType == DataTypes.Float8E4M3 || elementType == DataTypes.Float8E5M2)
        {
            var newType = new VectorType(DataTypes.UInt8, target.Lanes.ToArray());
            var input = context.GetArgumentValue(target, Devectorize.Input).AsTensor();
            input = Tensor.FromBytes(newType, input.BytesBuffer.ToArray(), input.Shape);
            var inputOrt = input.ToOrtTensor();

            foreach (var axis in target.Axes.Reverse())
            {
                inputOrt = inputOrt.Devectorize(axis);
            }

            var output = inputOrt.ToTensor();
            return Value.FromTensor(Tensor.FromBytes(elementType, output.BytesBuffer.ToArray(), output.Shape));
        }
        else
        {
            var input = context.GetOrtArgumentValue(target, Devectorize.Input);
            foreach (var axis in target.Axes.Reverse())
            {
                input = input.Devectorize(axis);
            }

            return Value.FromTensor(input.ToTensor());
        }
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Devectorize target)
    {
        var input = context.CheckArgumentType<IRType>(target, Devectorize.Input);

        return input switch
        {
            DistributedType d => Visit(context, target, d),
            TensorType t => Visit(context, target, t),
            AnyType => AnyType.Default,
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Devectorize target)
    {
        var inputType = context.GetArgumentType<IRType>(target, Devectorize.Input);
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Devectorize target)
    {
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
        };
    }

    private IRType Visit(ITypeInferenceContext context, Devectorize target, TensorType input)
    {
        if (target.Lanes.Any(x => x <= 0))
        {
            return new InvalidType("devectorize lane <= 0");
        }

        return TypeInference.DevectorizeType(input, target.Axes);
    }

    private IRType Visit(ITypeInferenceContext context, Devectorize target, DistributedType input)
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
