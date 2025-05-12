// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using DryIoc.ImTools;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Cast"/>.
/// </summary>
public class CastEvaluator : IEvaluator<Cast>, ITypeInferencer<Cast>, IOpPrinter<Cast>, ICostEvaluator<Cast>, IMetricEvaluator<Cast>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Cast cast)
    {
        var input = context.GetArgumentValue(cast, Cast.Input).AsTensor();
        var dimensions = input.Dimensions.ToArray();
        if (cast.NewType is VectorType vt && cast.PackAxes.Any())
        {
            var scale = 1f * vt.ElemType.SizeInBytes / ((VectorType)input.ElementType).ElemType.SizeInBytes;
            cast.PackAxes.ToArray().ForEach(a => dimensions[a] = (int)(dimensions[a] * scale));
        }

        return Value.FromTensor(input.CastTo(cast.NewType, cast.CastMode, dimensions));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Cast target)
    {
        var input = context.CheckArgumentType<IRType>(target, Cast.Input);
        return input switch
        {
            TensorType t => Visit(target, t),
            DistributedType d => Visit(target, d),
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public string Visit(IPrintOpContext context, Cast target)
    {
        return $"{CompilerServices.Print(target.NewType)}({context.GetArgument(target, Cast.Input)})";
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Cast target)
    {
        var input = context.GetArgumentType<IRType>(target, Cast.Input);
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(target.NewType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(target.NewType, 1),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Cast target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Cast.Input);
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) * 2,
        };
    }

    private IRType Visit(Cast target, TensorType input)
    {
        if (input.DType is VectorType vt)
        {
            if (target.PackAxes.Any(a => input.Shape[a] is { IsFixed: false }))
            {
                return new InvalidType("Pack axes must be fixed");
            }

            var scale = 1f * ((VectorType)target.NewType).ElemType.SizeInBytes / vt.ElemType.SizeInBytes;
            if (target.PackAxes.Any(a => input.Shape[a].FixedValue * scale % 1 != 0))
            {
                return new InvalidType("Pack axes must be divisible by scale");
            }

            // var outType = new VectorType(((VectorType)target.NewType).ElemType, vt.Lanes.Select(l => (int)(l / scale)).ToArray());
            var newShape = input.Shape.ToArray();
            target.PackAxes.ToArray().ForEach(a => newShape[a] = (int)(newShape[a].FixedValue * scale));
            return new TensorType(target.NewType, newShape);
        }

        return new TensorType(target.NewType, input.Shape);
    }

    private IRType Visit(Cast target, DistributedType inType)
    {
        var invalid = new InvalidType(inType.ToString());
        var outType = Visit(target, inType.TensorType);
        var ndsbp = new SBP[inType.TensorType.Shape.Rank];
        var shape = CompilerServices.GetMaxShape(inType.TensorType.Shape);
        for (int i = 0; i < ndsbp.Length; i++)
        {
            if (inType.AxisPolices[i] is SBPPartial)
            {
                return invalid;
            }

            if (inType.AxisPolices[i] is SBPSplit split && inType.TensorType.DType is VectorType vtIn && outType is TensorType ttOut && ttOut.DType is VectorType vtOut)
            {
                if (vtIn.ElemType != vtOut.ElemType)
                {
                    var divisor = split.Axes.Select(a => inType.Placement.Hierarchy[a]).Aggregate(1, (a, b) => a * b);
                    if (shape[i] % divisor != 0)
                    {
                        return invalid;
                    }
                }
            }

            ndsbp[i] = inType.AxisPolices[i];
        }

        return new DistributedType((TensorType)outType, ndsbp, inType.Placement);
    }
}
