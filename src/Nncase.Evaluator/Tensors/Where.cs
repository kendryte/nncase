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
        if (where.IsTfWhere)
        {
            var condTensor = context.GetArgumentValueAsTensor<bool>(where, Where.Cond);
            if (condTensor.Rank > 1)
            {
                throw new NotImplementedException();
            }

            var result = condTensor.Select((b, i) => (b, i)).Where(t => t.b).Select(t => (long)t.i).ToArray();
            return Value.FromTensor(Tensor.From<long>(result, [result.Length, condTensor.Rank]));
        }

        var cond = context.GetArgumentValue(where, Where.Cond).AsTensor();
        var x = context.GetArgumentValue(where, Where.X).AsTensor();
        var y = context.GetArgumentValue(where, Where.Y).AsTensor();
        var condType = cond.ElementType;
        var xType = x.ElementType;
        var yType = y.ElementType;
        if (xType is VectorType { ElemType: DataType xElemType } xVType && xElemType != DataTypes.Float32)
        {
            var interType = new VectorType(DataTypes.Float32, xVType.Lanes);
            x = Nncase.IR.F.Tensors.Cast(x, interType).Evaluate().AsTensor();
        }
        else if (xType.IsFloat() && xType is not VectorType && xType != DataTypes.Float32)
        {
            x = x.CastTo(DataTypes.Float32);
        }

        if (yType is VectorType { ElemType: DataType elemType } yVType && elemType != DataTypes.Float32)
        {
            var interType = new VectorType(DataTypes.Float32, yVType.Lanes);
            y = Nncase.IR.F.Tensors.Cast(y, interType).Evaluate().AsTensor();
        }
        else if (yType.IsFloat() && yType is not VectorType && yType != DataTypes.Float32)
        {
            y = y.CastTo(DataTypes.Float32);
        }

        var condOrt = cond.ToOrtTensor();
        var xOrt = x.ToOrtTensor();
        var yOrt = y.ToOrtTensor();
        var condLaneNum = condType is MaskVectorType vt1 ? 1 : 0;
        var xLaneNum = xType is VectorType vt2 ? vt2.Lanes.Count : 0;
        var yLaneNum = yType is VectorType vt3 ? vt3.Lanes.Count : 0;
        var maxLaneSize = System.Math.Max(System.Math.Max(condLaneNum, xLaneNum), yLaneNum);
        if (condLaneNum < maxLaneSize)
        {
            condOrt = OrtKI.Unsqueeze(condOrt, Enumerable.Range(-maxLaneSize, maxLaneSize - condLaneNum).Select(a => (long)a).ToArray());
        }

        if (xLaneNum < maxLaneSize)
        {
            xOrt = OrtKI.Unsqueeze(xOrt, Enumerable.Range(-maxLaneSize, maxLaneSize - xLaneNum).Select(a => (long)a).ToArray());
        }

        if (yLaneNum < maxLaneSize)
        {
            yOrt = OrtKI.Unsqueeze(yOrt, Enumerable.Range(-maxLaneSize, maxLaneSize - yLaneNum).Select(a => (long)a).ToArray());
        }

        if (maxLaneSize > 0)
        {
            var output = OrtKI.Where(condOrt, xOrt, yOrt);
            var outShape = context.Evaluate(context.CurrentCall.CheckedShape).AsTensor().ToArray<long>();
            return Value.FromTensor(Tensor.FromBytes(context.CurrentCall.CheckedDataType, output.BytesBuffer.ToArray(), outShape));
        }

        return Value.FromTensor(OrtKI.Where(condOrt, xOrt, yOrt).ToTensor().CastTo(context.CurrentCall.CheckedDataType));
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

        // FIXME: remove this when ntt::where is ready
        if (cond.DType is VectorType)
        {
            return new InvalidType("cond can't be vector type");
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
            var policyCond = i < padCond ? null : cond.AxisPolicies[i - padCond];
            var policyX = i < padX ? null : x.AxisPolicies[i - padX];
            var policyY = i < padY ? null : y.AxisPolicies[i - padY];

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
                if (b[axis - padB] is { IsFixed: false } || (b[axis - padB] is { IsFixed: true, FixedValue: var fb } && fb != 1))
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
                if (a[axis - padA] is { IsFixed: false } || (a[axis - padA] is { IsFixed: true, FixedValue: var fa } && fa != 1))
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
