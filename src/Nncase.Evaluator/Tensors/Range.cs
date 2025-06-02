// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Shapes;
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
        var begin = (Expr)context.GetArgument(target, Range.Begin);
        var end = (Expr)context.GetArgument(target, Range.End);
        var step = (Expr)context.GetArgument(target, Range.Step);
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

        switch (begin, end, step)
        {
            case (TensorConst beginConst, TensorConst endConst, TensorConst stepConst):
                var dim1 = (long)MathF.Ceiling((endConst.Value.ToScalar<float>() - beginConst.Value.ToScalar<float>()) / stepConst.Value.ToScalar<float>());
                return new TensorType(dType, new RankedShape(dim1));
            case (Call { Target: AsTensor } tBegin, Call { Target: AsTensor } tEnd, TensorConst stc) when stc.Value.ToScalar<long>() == 1:
                var dim2 = (Dimension)tEnd[AsTensor.Input] - (Dimension)tBegin[AsTensor.Input];
                return new TensorType(dType, new RankedShape(dim2));
            case (TensorConst beginConst, Call { Target: AsTensor } tEnd, TensorConst stc) when stc.Value.ToScalar<long>() == 1:
                var dim3 = (Dimension)tEnd[AsTensor.Input] - (Dimension)beginConst.Value.ToScalar<long>();
                return new TensorType(dType, new RankedShape(dim3));
            case (Call { Target: AsTensor } tBegin, TensorConst endConst, TensorConst stc) when stc.Value.ToScalar<long>() == 1:
                var dim4 = (Dimension)endConst.Value.ToScalar<long>() - (Dimension)tBegin[AsTensor.Input];
                return new TensorType(dType, new RankedShape(dim4));
            default:
                var dim5 = IR.F.Tensors.Cast(IR.F.Math.CeilDiv(end - begin, step), DataTypes.Int64).AsDim();
                return new TensorType(dType, new RankedShape(dim5));
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
