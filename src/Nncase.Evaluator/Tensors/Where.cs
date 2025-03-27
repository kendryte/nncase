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
public class WhereEvaluator : IEvaluator<Where>, ITypeInferencer<Where>, ICostEvaluator<Where>, IMetricEvaluator<Where>
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
            return new TensorType(DataTypes.Int64, Shape.Unknown(cond.Shape.Rank));
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

        // if (cond.TensorType.Shape != targetType.Shape)
        // {
        //     return invalid;
        // }
        var padCond = targetType.Shape.Rank - cond.TensorType.Shape.Rank;
        var padX = targetType.Shape.Rank - x.TensorType.Shape.Rank;
        var padY = targetType.Shape.Rank - y.TensorType.Shape.Rank;

        var ndsbp = new SBP[targetType.Shape.Rank];
        for (int i = 0; i < ndsbp.Length; i++)
        {
            var policyCond = i < padCond ? null : cond.AxisPolices[i - padCond];
            var policyX = i < padX ? null : x.AxisPolices[i - padX];
            var policyY = i < padY ? null : y.AxisPolices[i - padY];

            SBP? policyOut;
            switch (policyCond, policyX, policyY)
            {
                case (null, _, _):
                    policyOut = CheckSBP(policyX, policyY, x.TensorType.Shape, y.TensorType.Shape, padX, padY, i);
                    break;
                case (_, null, _):
                    policyOut = CheckSBP(policyCond, policyY, cond.TensorType.Shape, y.TensorType.Shape, padCond, padY, i);
                    break;
                case (_, _, null):
                    policyOut = CheckSBP(policyCond, policyX, cond.TensorType.Shape, x.TensorType.Shape, padCond, padX, i);
                    break;
                default:
                    var policyXY = CheckSBP(policyX, policyY, x.TensorType.Shape, y.TensorType.Shape, padX, padY, i);
                    if (policyXY is null)
                    {
                        return invalid;
                    }

                    var xyType = (TensorType)TypeInference.BroadcastType(x.TensorType.DType, x.TensorType, y.TensorType);
                    var padXY = targetType.Shape.Rank - xyType.Shape.Rank;
                    policyOut = CheckSBP(policyCond, policyXY, cond.TensorType.Shape, xyType.Shape, padCond, padXY, i);
                    break;
            }

            if (policyOut is null)
            {
                return invalid;
            }

            ndsbp[i] = policyOut!;
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

    private SBP? CheckSBP(SBP? policyA, SBP? policyB, Shape a, Shape b, int padA, int padB, int axis)
    {
        SBP? ret;
        switch (policyA, policyB)
        {
            case (null, null):
                ret = null;
                break;
            case (null, _):
                ret = policyB!;
                break;
            case (_, null):
                ret = policyA!;
                break;
            case (SBPSplit sa, SBPSplit sb):
                if (sa.Axes != sb.Axes)
                {
                    ret = null;
                }
                else
                {
                    ret = sa;
                }

                break;
            case (SBPSplit sa, SBPBroadCast):
                // invalid (S, B) if B is not broacast
                if (b[axis - padB] != 1)
                {
                    ret = null;
                }
                else
                {
                    ret = sa;
                }

                break;
            case (SBPBroadCast, SBPSplit sb):
                // invalid (B, S) if A is not broacast
                if (a[axis - padA] != 1)
                {
                    ret = null;
                }
                else
                {
                    ret = sb;
                }

                break;
            case (SBPBroadCast, SBPBroadCast):
                ret = SBP.B;
                break;
            default:
                ret = null;
                break;
        }

        return ret;
    }
}
