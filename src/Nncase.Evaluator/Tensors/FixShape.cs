// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using DryIoc.ImTools;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="FixShape"/>.
/// </summary>
public class FixShapeEvaluator : IEvaluator<FixShape>, ITypeInferencer<FixShape>, ICostEvaluator<FixShape>, IShapeEvaluator<FixShape>, IMetricEvaluator<FixShape>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, FixShape fixShape)
    {
        return context.GetArgumentValue(fixShape, FixShape.Input);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, FixShape target)
    {
        var input = context.CheckArgumentType<TensorType>(target, FixShape.Input);
        var shape = ((TensorConst)context.GetArgument(target, FixShape.Shape)).Value.ToArray<int>();
        return input with { Shape = shape };
    }

    public Cost Visit(ICostEvaluateContext context, FixShape target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, FixShape.Input);
        var outputType = context.GetReturnType<TensorType>();
        return new Cost()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(inputType, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Mul)),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, FixShape target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, FixShape.Input);
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, FixShape target) => context.GetArgument(target, FixShape.Shape);
}
