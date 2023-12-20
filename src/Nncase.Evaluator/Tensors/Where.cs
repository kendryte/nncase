// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Where"/>.
/// </summary>
public class WhereEvaluator : IEvaluator<Where>, ITypeInferencer<Where>, ICostEvaluator<Where>, IShapeEvaluator<Where>, IMetricEvaluator<Where>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Where where)
    {
        var xt = context.GetArgumentValueAsTensor(where, Where.X);
        var yt = context.GetArgumentValueAsTensor(where, Where.Y);
        if (where.IsTfWhere)
        {
            var condTensor = context.GetArgumentValueAsTensor<bool>(where, Where.Cond);
            if (condTensor.Rank > 1)
            {
                throw new NotImplementedException();
            }

            var result = condTensor.Select((b, i) => (b, i)).Where(t => t.b).Select(t => (long)t.i).ToArray();
            return Value.FromTensor(Tensor.From<long>(result, new Shape(result.Length, condTensor.Rank)));
        }

        var cond = context.GetOrtArgumentValue(where, Where.Cond);
        var x = context.GetOrtArgumentValue(where, Where.X);
        var y = context.GetOrtArgumentValue(where, Where.Y);
        return OrtKI.Where(cond, x, y).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Where target)
    {
        var cond = context.CheckArgumentType<IRType>(target, Where.Cond);
        var x = context.CheckArgumentType<IRType>(target, Where.X);
        var y = context.CheckArgumentType<IRType>(target, Where.Y);

        return (cond, x, y) switch
        {
            (DistributedType a, DistributedType b, DistributedType c) => Visit(a, b, c, target),
            (TensorType a, TensorType b, TensorType c) => Visit(a, b, c, target),
            _ => new InvalidType(cond.GetType().ToString()),
        };
    }

    public IRType Visit(TensorType cond, TensorType x, TensorType y, Where target)
    {
        if (target.IsTfWhere)
        {
            return new TensorType(DataTypes.Int64, new Shape(Dimension.Unknown, cond.Shape.Rank));
        }

        return TypeInference.BroadcastType(x.DType, cond, x, y);
    }

    public IRType Visit(DistributedType cond, DistributedType x, DistributedType y, Where target)
    {
        var invalid = new InvalidType($"{cond}, {x}, {y} not support");
        if (cond.Placement != x.Placement || x.Placement != y.Placement)
        {
            return invalid;
        }

        if (target.IsTfWhere)
        {
            return invalid;
        }

        var targetType = (TensorType)TypeInference.BroadcastType(x.TensorType.DType, cond.TensorType, x.TensorType, y.TensorType);
        if (cond.TensorType.Shape != targetType.Shape)
        {
            return invalid;
        }

        var ndsbp = new SBP[cond.Placement.Rank];

        for (int i = 0; i < cond.Placement.Rank; i++)
        {
            switch (cond.NdSBP[i], x.NdSBP[i], y.NdSBP[i])
            {
                case (SBPSplit { Axis: int ic }, SBPSplit { Axis: int }, SBPSplit { Axis: int }):
                    ndsbp[i] = SBP.S(ic);
                    break;
                case (SBPSplit { Axis: int ic }, SBPBroadCast, SBPSplit { Axis: int }):
                    ndsbp[i] = SBP.S(ic);
                    break;
                case (SBPSplit { Axis: int ic }, SBPSplit { Axis: int }, SBPBroadCast):
                    ndsbp[i] = SBP.S(ic);
                    break;
                case (SBPSplit { Axis: int ic }, SBPBroadCast, SBPBroadCast):
                    ndsbp[i] = SBP.S(ic);
                    break;
                case (SBPBroadCast, SBPBroadCast, SBPBroadCast):
                    ndsbp[i] = SBP.B;
                    break;
                default:
                    return invalid;
            }
        }

        return new DistributedType(targetType, ndsbp, cond.Placement);
    }

    public Cost Visit(ICostEvaluateContext context, Where target)
    {
        var cond = context.GetArgumentType<IRType>(target, Where.Cond);
        var x = context.GetArgumentType<IRType>(target, Where.X);
        var y = context.GetArgumentType<IRType>(target, Where.Y);
        var ret = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(cond, x, y),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(cond, CostUtility.GetCPUCyclesOfCompare()),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Where target)
    {
        var x = context.GetArgumentShape(target, Where.X);
        if (target.IsTfWhere)
        {
            var condValue = context.GetArgument(target, Where.Cond);
            var condShape = context.GetArgumentShape(target, Where.Cond);
            if (condValue.CheckedShape.Rank == 1)
            {
                return IR.F.Tensors.Stack(new IR.Tuple(new[] { condShape[0], x[0] }), 0);
            }

            throw new NotImplementedException();
        }

        var y = context.GetArgumentShape(target, Where.Y);
        var cond = context.GetArgumentShape(target, Where.Cond);
        return ShapeExprUtility.BroadcastShape(x, y, cond);
    }

    public Metric Visit(IMetricEvaluateContext context, Where target)
    {
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(returnType),
        };
    }

    private bool IsTFWhere(TensorType x, TensorType y)
    {
        return x.Shape[0] == 0 && y.Shape[0] == 0 && x.DType == DataTypes.Float32;
    }
}
