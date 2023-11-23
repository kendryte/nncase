// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Cast"/>.
/// </summary>
public class CastEvaluator : IEvaluator<Cast>, ITypeInferencer<Cast>, IOpPrinter<Cast>, ICostEvaluator<Cast>, IShapeEvaluator<Cast>, IMetricEvaluator<Cast>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Cast cast)
    {
        var input = context.GetArgumentValue(cast, Cast.Input).AsTensor();
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
    public string Visit(IIRPrinterContext context, Cast target, bool iLmode)
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

    public Expr Visit(IShapeEvaluateContext context, Cast target) => context.GetArgumentShape(target, Cast.Input);

    private IRType Visit(Cast target, TensorType input)
    {
        return new TensorType(target.NewType, input.Shape);
    }

    private IRType Visit(Cast target, DistributedType inType)
    {
        var invalid = new InvalidType(inType.ToString());
        var ndsbp = new SBP[inType.Placement.Rank];
        for (int i = 0; i < inType.Placement.Rank; i++)
        {
            if (inType.NdSBP[i] is SBPPartialSum)
            {
                return invalid;
            }

            ndsbp[i] = inType.NdSBP[i];
        }

        return new DistributedType(new TensorType(target.NewType, inType.TensorType.Shape), ndsbp, inType.Placement);
    }
}
