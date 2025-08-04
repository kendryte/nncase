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
        if (cast.NewType is VectorType vt && !cast.VectorizeAxes.IsDefaultOrEmpty)
        {
            if (cast.VectorizeAxes.Count > 1)
            {
                throw new NotSupportedException("Vectorize axes must be one");
            }

            input = IR.F.Tensors.Devectorize(input, ((VectorType)input.ElementType).Lanes.ToArray(), cast.VectorizeAxes.ToArray()).Evaluate().AsTensor();
            input = input.CastTo(vt.ElemType);
            input = IR.F.Tensors.Vectorize(input, vt.Lanes.ToArray(), cast.VectorizeAxes.ToArray()).Evaluate().AsTensor();
            return Value.FromTensor(input);
        }

        return Value.FromTensor(input.CastTo(cast.NewType, cast.CastMode));
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
            if (!target.VectorizeAxes.IsDefaultOrEmpty && target.VectorizeAxes.Any(a => input.Shape[a] is { IsFixed: false }))
            {
                return new InvalidType("Vectorize axes must be fixed");
            }

            var scale = 1f;
            var newShape = input.Shape.ToArray();
            if (!target.VectorizeAxes.IsDefaultOrEmpty)
            {
                scale = 1f * ((VectorType)target.NewType).ElemType.SizeInBytes / vt.ElemType.SizeInBytes;
                if (target.VectorizeAxes.Any(a => input.Shape[a].FixedValue * scale % 1 != 0))
                {
                    return new InvalidType("Vectorize axes must be divisible by scale");
                }

                target.VectorizeAxes.ToArray().ForEach(a => newShape[a] = (int)(newShape[a].FixedValue * scale));
            }

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
            if (inType.AxisPolicies[i] is SBPPartial)
            {
                return invalid;
            }

            if (inType.AxisPolicies[i] is SBPSplit split && inType.TensorType.DType is VectorType vtIn && outType is TensorType ttOut && ttOut.DType is VectorType vtOut)
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

            ndsbp[i] = inType.AxisPolicies[i];
        }

        return new DistributedType((TensorType)outType, ndsbp, inType.Placement);
    }
}
