// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.Utilities;
using OrtKISharp;
using Range = Nncase.IR.Tensors.Range;
using static Nncase.Utilities.ShapeExprUtility;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Range"/>.
/// </summary>
public class RangeEvaluator : IEvaluator<Range>, ITypeInferencer<Range>, ICostEvaluator<Range>, IMetricEvaluator<Range>, IShapeEvaluator<Range>
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
        if (begin is TensorConst beginValue
             && end is TensorConst endValue
             && step is TensorConst stepValue)
        {
            var dType = begin.CheckedDataType;
            if (!(begin.CheckedDataType == end.CheckedDataType &&
                  end.CheckedDataType == step.CheckedDataType))
            {
                return new InvalidType($"Range Begin End Step must be same type, " +
                                       $"but get begin:{begin.CheckedDataType}," +
                                       $"end:{end.CheckedDataType}," +
                                       $"step:{step.CheckedDataType}");
            }

            return new TensorType(
                dType,
                new Shape((beginValue.Value.ToScalar<int>() + endValue.Value.ToScalar<int>()) /
                          stepValue.Value.ToScalar<int>()));
        }
        else
        {
            DataType dt;
            if (begin.CheckedType is TensorType beginCheckedType)
            {
                dt = beginCheckedType.DType;
            }
            else if (end.CheckedType is TensorType endCheckedType)
            {
                dt = endCheckedType.DType;
            }
            else
            {
                return new InvalidType("DataType is unknown");
            }

            return new TensorType(dt, new Shape(Dimension.Unknown));
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

    public Expr Visit(IShapeEvaluateContext context, Range target)
    {
        var begin = context.GetArgument(target, Range.Begin);
        var end = context.GetArgument(target, Range.End);
        var step = context.GetArgument(target, Range.Step);
        return ShapeExprUtility.StackOne((end - begin) / step);
    }
}
