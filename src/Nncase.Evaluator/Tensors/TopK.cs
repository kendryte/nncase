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
        var xArg = context.CheckArgumentType<IRType>(target, TopK.X);
        var kArg = context.CheckArgumentType<IRType>(target, TopK.K);

        return (xArg, kArg) switch
        {
            (TensorType xt, TensorType kt) => VisitTensor(context, target, xt, kt),
            (DistributedType dx, TensorType kt) => VisitDistributed(context, target, dx, kt, null),
            (DistributedType dx, DistributedType dk) => VisitDistributed(context, target, dx, dk.TensorType, dk),
            (TensorType xt, DistributedType dk) when dk.AxisPolicies.All(p => p is SBPBroadCast) => VisitTensor(context, target, xt, dk.TensorType),
            _ => new InvalidType($"{xArg}, {kArg}"),
        };
    }

    public Cost Visit(ICostEvaluateContext context, TopK target)
    {
        var x = context.GetArgumentType<IRType>(target, TopK.X);
        var k = context.GetArgumentType<IRType>(target, TopK.K);
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

    private IRType VisitTensor(ITypeInferenceContext context, TopK target, TensorType x, TensorType k)
    {
        if (x.Shape is not RankedShape xShape
            || k.Shape is not RankedShape)
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
            var shapeArr = xShape.ToArray();
            shapeArr[axis] = kConst.Value.ToArray<int>()[0];
            shape = new RankedShape(shapeArr);
        }
        else
        {
            shape = Shape.Unknown(xShape.Rank - 1);
        }

        return new TupleType(new[] { x with { Shape = shape }, new TensorType(DataTypes.Int64, shape) });
    }

    private IRType VisitDistributed(ITypeInferenceContext context, TopK target, DistributedType x, TensorType kTensor, DistributedType? kDistributed)
    {
        // K must be broadcast-only (or plain tensor).
        if (kTensor.DType != DataTypes.Int64)
        {
            return new InvalidType("TopK K need int64");
        }

        if (kDistributed != null)
        {
            if (kDistributed.Placement != x.Placement || kDistributed.AxisPolicies.Any(policy => policy is not SBPBroadCast))
            {
                return new InvalidType("TopK only supports broadcast K in distributed mode");
            }
        }

        var axisExpr = context.GetArgument(target, TopK.Axis) as TensorConst;
        if (axisExpr is null)
        {
            return new InvalidType("Distributed TopK requires constant axis");
        }

        var axis = Util.PositiveIndex(axisExpr.Value.ToScalar<int>(), x.TensorType);

        // Axis policy must be broadcast; other dims can be broadcast or split (no partial allowed).
        for (int i = 0; i < x.AxisPolicies.Count; i++)
        {
            var policy = x.AxisPolicies[i];
            if (policy is SBPPartial)
            {
                return new InvalidType("TopK doesn't support partial sbp in distributed mode");
            }

            if (i == axis)
            {
                if (policy is not SBPBroadCast)
                {
                    return new InvalidType("TopK axis must be broadcast when distributed");
                }
            }
            else if (policy is not SBPBroadCast && policy is not SBPSplit)
            {
                return new InvalidType("Unsupported SBP policy for distributed TopK");
            }
        }

        if (VisitTensor(context, target, x.TensorType, kTensor) is not TupleType tuple
            || tuple.Count != 2
            || tuple[0] is not TensorType values
            || tuple[1] is not TensorType indices)
        {
            return new InvalidType("TopK tensor inference failed");
        }

        var valuesD = new DistributedType(values, x.AxisPolicies, x.Placement);
        var indicesD = new DistributedType(indices, x.AxisPolicies, x.Placement);
        return new TupleType(new IRType[] { valuesD, indicesD });
    }
}
