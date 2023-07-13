// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.TIR;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="TopK"/>.
/// </summary>
public class TopKEvaluator : IEvaluator<TopK>, ITypeInferencer<TopK>, ICostEvaluator<TopK>, IMetricEvaluator<TopK>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, TopK topK)
    {
        var x = context.GetOrtArgumentValue(topK, TopK.X);
        var k = context.GetOrtArgumentValue(topK, TopK.K);
        var axis = context.GetArgumentValueAsScalar<long>(topK, TopK.Axis);
        var largest = context.GetArgumentValueAsScalar<long>(topK, TopK.Largest);
        var sorted = context.GetArgumentValueAsScalar<long>(topK, TopK.Sorted);
        return new TupleValue(OrtKI.TopK(x, k, axis, largest, sorted).Select(x => x.ToValue()).ToArray());
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, TopK target)
    {
        var input = context.CheckArgumentType<TensorType>(target, TopK.X);
        var repeat = context.CheckArgumentType<TensorType>(target, TopK.K);
        return Visit(context, target, input, repeat);
    }

    public Cost Visit(ICostEvaluateContext context, TopK target)
    {
        var x = context.GetArgumentType<TensorType>(target, TopK.X);
        var k = context.GetArgumentType<TensorType>(target, TopK.K);
        var outputType = context.GetReturnType<TupleType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(x) + CostUtility.GetMemoryAccess(k),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, TopK target)
    {
        var x = context.GetArgumentType<TensorType>(target, TopK.X);
        var outputType = context.GetReturnType<TupleType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(x) + CostUtility.GetMemoryAccess(outputType),
        };
    }

    private IRType Visit(ITypeInferenceContext context, TopK target, TensorType x, TensorType k)
    {
        if (x.Shape.IsUnranked || k.Shape.IsUnranked)
        {
            return new TupleType(new[] { x, new TensorType(DataTypes.Int64, Shape.Unranked) });
        }

        if (k.DType != DataTypes.Int64)
        {
            return new InvalidType("TopK K need int64");
        }

        // x: [a_1, a_2, ..., a_n, r]
        Shape? shape;

        // [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
        if (context.GetArgument(target, TopK.Axis) is TensorConst axisConst
            && context.GetArgument(target, TopK.K) is TensorConst kConst)
        {
            var axis = Util.PositiveIndex(axisConst.Value.ToScalar<int>(), x);
            var shapeArr = x.Shape.ToArray();
            shapeArr[axis] = kConst.Value.ToArray<int>()[0];
            shape = new Shape(shapeArr);
        }
        else
        {
            shape = Shape.Unknown(x.Shape.Rank - 1);
        }

        return new TupleType(new[] { x with { Shape = shape }, new TensorType(DataTypes.Int64, shape) });
    }
}
