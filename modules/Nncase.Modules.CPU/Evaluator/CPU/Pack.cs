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

public sealed class PackEvaluator : ITypeInferencer<Pack>, ICostEvaluator<Pack>, IEvaluator<Pack>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Pack target)
    {
        var input = context.GetOrtArgumentValue(target, Pack.Input);
        var paddedDim = MathUtility.AlignUp(input.Shape[target.Axis], target.Lanes);
        var paddings = Enumerable.Repeat(0L, input.Rank + target.Axis).Append(paddedDim - input.Shape[target.Axis])
            .Concat(Enumerable.Repeat(0L, input.Rank - target.Axis - 1)).ToArray();
        var padded = OrtKI.Pad(input, paddings, 0, "constant");
        var dividedShape = input.Shape.Take(target.Axis).Concat(new[] { paddedDim / target.Lanes, target.Lanes }).Concat(input.Shape.Skip(target.Axis + 1)).ToArray();
        var perm = Enumerable.Range(0, target.Axis + 1).Concat(Enumerable.Range(target.Axis + 2, dividedShape.Length - (target.Axis + 2))).Cast<long>().ToArray();
        var tp = OrtKI.Transpose(OrtKI.Reshape(padded, dividedShape, 0), perm);
        return tp.ToValue();
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
        return TypeInference.PackType(input, target.Lanes, target.Axis);
    }

    private IRType Visit(ITypeInferenceContext context, Pack target, DistributedType input)
    {
        if (Visit(context, target, input.TensorType) is not TensorType tensorType)
        {
            throw new InvalidOperationException();
        }

        return new DistributedType(tensorType, input.NdSBP, input.Placement);
    }
}
