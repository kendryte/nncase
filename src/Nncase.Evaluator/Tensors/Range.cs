// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.Utilities.ShapeExprUtility;
using Range = Nncase.IR.Tensors.Range;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Range"/>.
/// </summary>
public class RangeEvaluator : IEvaluator<Range>, ITypeInferencer<Range>, ICostEvaluator<Range>, IMetricEvaluator<Range>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Range range)
    {
        var begin = context.GetOrtArgumentValue(range, Range.Begin);
        var end = context.GetOrtArgumentValue(range, Range.End);
        var step = context.GetOrtArgumentValue(range, Range.Step);
        return OrtKI.Range(begin, end, step).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Range target)
    {
        var begin = context.GetArgument(target, Range.Begin);
        var end = context.GetArgument(target, Range.End);
        var step = context.GetArgument(target, Range.Step);
        var dType = begin.CheckedDataType;
        if (!(begin.CheckedDataType == end.CheckedDataType &&
              end.CheckedDataType == step.CheckedDataType))
        {
            return new InvalidType($"Range Begin End Step must be same type, " +
                                   $"but get begin:{begin.CheckedDataType}," +
                                   $"end:{end.CheckedDataType}," +
                                   $"step:{step.CheckedDataType}");
        }

        if (!begin.CheckedShape.IsScalar)
        {
            return new InvalidType($"Range Begin must be scalar, but get {begin.CheckedShape}");
        }

        if (begin is TensorConst beginConst && end is TensorConst endConst && step is TensorConst stepConst)
        {
            var dim = (long)MathF.Ceiling(beginConst.Value.ToScalar<float>() + endConst.Value.ToScalar<float>()) / stepConst.Value.ToScalar<float>();
            return new TensorType(dType, new Shape(dim));
        }
        else
        {
            var dim = IR.F.Tensors.Cast(IR.F.Math.CeilDiv(begin + end, step), DataTypes.Int64);
            return new TensorType(dType, new Shape(dim));
        }
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Range target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = 0,
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Range target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(ret),
        };
    }
}
