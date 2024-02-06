// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Ncnn;
using Nncase.IR.Tensors;
using OrtKISharp;
using static Nncase.Evaluator.EvaluatorUtil;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnReduction"/>.
/// </summary>
public class NcnnReductionEvaluator : IEvaluator<NcnnReduction>, ITypeInferencer<NcnnReduction>, ICostEvaluator<NcnnReduction>, IShapeEvaluator<NcnnReduction>, IMetricEvaluator<NcnnReduction>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnReduction reduction)
    {
        var input = context.GetOrtArgumentValue(reduction, NcnnReduction.Input);
        var axes = reduction.Args.Axes;
        var keepDims = reduction.Args.Keepdims;

        return (reduction.Args.OpType switch
        {
            0 => OrtKI.ReduceSum(input, axes, keepDims, 0),
            3 => OrtKI.ReduceMean(input, axes.Cast<long>().ToArray(), keepDims),
            4 => OrtKI.ReduceMax(input, axes.Cast<long>().ToArray(), keepDims),
            5 => OrtKI.ReduceMin(input, axes.Cast<long>().ToArray(), keepDims),
            6 => OrtKI.ReduceProd(input, axes.Cast<long>().ToArray(), keepDims),
            _ => throw
                new NotImplementedException($"Reduction opType {reduction.Args.OpType} is not supported."),
        }).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnReduction target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnReduction.Input);
        return Visit(target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnReduction target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnReduction.Input);
        var returnType = context.GetReturnType<TensorType>();
        var rF = MetricUtility.GetFLOPs(returnType);
        var iF = MetricUtility.GetFLOPs(inputType);
        var inner = iF / rF;
        _ = iF / inner;
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(returnType),
            [MetricFactorNames.FLOPs] = iF,
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnReduction target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnReduction.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(inputType) * (MetricUtility.DivFLOPs + MetricUtility.ExpFLOPs + MetricUtility.MulFLOPs + MetricUtility.SubFLOPs + MetricUtility.AddFLOPs + (MetricUtility.CmpFLOPs * 2)),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnReduction target) => context.GetArgumentShape(target, NcnnReduction.Input);

    private IRType Visit(NcnnReduction reduction, TensorType input)
    {
        var newInput = new TensorType(input.DType, input.Shape.InsertAndClone(0, 1));
        var axis = reduction.Args.Axes;

        var newAxis = axis.Select(x => x > 0 ? x + 1 : x).ToArray();

        var output_ = TypeInference.ReduceType(newInput, reduction.Args.Keepdims, newAxis);
        if (output_ is TensorType t)
        {
            var newShape = t.Shape.ToArray();
            return new TensorType(t.DType, newShape[1..]);
        }

        return output_;
    }
}
