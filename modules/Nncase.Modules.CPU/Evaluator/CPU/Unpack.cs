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
using Nncase.IR.CPU;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.CPU;

public sealed class UnpackEvaluator : ITypeInferencer<Unpack>, ICostEvaluator<Unpack>, IEvaluator<Unpack>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Unpack target)
    {
        var input = context.GetOrtArgumentValue(target, Unpack.Input);
        foreach (var axis in target.Axes.Reverse())
        {
            input = input.Unpack(axis);
        }

        return Value.FromTensor(input.ToTensor());
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
        return TypeInference.UnpackType(input, target.Axes);
    }

    private IRType Visit(ITypeInferenceContext context, Unpack target, DistributedType input)
    {
        if (Visit(context, target, input.TensorType) is not TensorType tensorType)
        {
            throw new InvalidOperationException();
        }

        return new DistributedType(tensorType, input.NdSBP, input.Placement);
    }
}
