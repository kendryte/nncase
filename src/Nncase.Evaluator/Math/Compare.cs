// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Compare"/>.
/// </summary>
public class CompareEvaluator : IEvaluator<Compare>, ITypeInferencer<Compare>, ICostEvaluator<Compare>, IOpPrinter<Compare>, IMetricEvaluator<Compare>
{
    public static IRType CheckSBP(TensorType tensorType, DistributedType a, DistributedType b)
    {
        // assume broadcast shapes are left algin
        var padA = tensorType.Shape.Rank - a.TensorType.Shape.Rank;
        var padB = tensorType.Shape.Rank - b.TensorType.Shape.Rank;
        var ndsbp = new SBP[tensorType.Shape.Rank];
        for (int i = 0; i < ndsbp.Length; i++)
        {
            var policyA = i < padA ? null : a.AxisPolices[i - padA];
            var policyB = i < padB ? null : b.AxisPolices[i - padB];
            switch (policyA, policyB)
            {
                case (null, _):
                    ndsbp[i] = policyB!;
                    break;
                case (_, null):
                    ndsbp[i] = policyA!;
                    break;
                case (SBPSplit sa, SBPSplit sb):
                    if (sa.Axes != sb.Axes)
                    {
                        return new InvalidType($"lhs rhs sbp at {i} not equal");
                    }

                    ndsbp[i] = sa;
                    break;
                case (SBPSplit sa, SBPBroadCast):
                    // invalid (S, B) if B is not broacast
                    if (b.TensorType.Shape[i - padB] != 1)
                    {
                        return new InvalidType($"lhs rhs sbp at {i} not broadcast");
                    }

                    ndsbp[i] = sa;
                    break;
                case (SBPBroadCast, SBPSplit sb):
                    // invalid (B, S) if A is not broacast
                    if (a.TensorType.Shape[i - padA] != 1)
                    {
                        return new InvalidType($"lhs rhs sbp at {i} not broadcast");
                    }

                    ndsbp[i] = sb;
                    break;
                case (SBPBroadCast, SBPBroadCast):
                    ndsbp[i] = SBP.B;
                    break;
                default:
                    return new InvalidType("not support compare sbp.");
            }
        }

        return new DistributedType(tensorType, ndsbp, a.Placement);
    }

    /// <inheritdoc />
    public IValue Visit(IEvaluateContext context, Compare target)
    {
        var lhs = context.GetArgumentValueAsTensor(target, Compare.Lhs);
        var rhs = context.GetArgumentValueAsTensor(target, Compare.Rhs);
        if (lhs.Shape.IsScalar && rhs.Shape.IsScalar && lhs.ElementType == DataTypes.Int32 && rhs.ElementType == DataTypes.Int32)
        {
            return Value.FromTensor(Tensor.FromScalar(Compute(target.CompareOp, lhs.ToScalar<int>(), rhs.ToScalar<int>())));
        }

        var a = context.GetOrtArgumentValue(target, Compare.Lhs);
        var b = context.GetOrtArgumentValue(target, Compare.Rhs);
        return target.CompareOp switch
        {
            CompareOp.Equal => OrtKI.Equal(a, b).ToValue(),
            CompareOp.LowerOrEqual => OrtKI.LessOrEqual(a, b).ToValue(),
            CompareOp.GreaterOrEqual => OrtKI.GreaterOrEqual(a, b).ToValue(),
            CompareOp.GreaterThan => OrtKI.Greater(a, b).ToValue(),
            CompareOp.LowerThan => OrtKI.Less(a, b).ToValue(),
            CompareOp.NotEqual => OrtKI.Not(OrtKI.Equal(a, b)).ToValue(),
            _ => throw new ArgumentOutOfRangeException(target.CompareOp.ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Compare target)
    {
        var lhsType = context.GetArgumentType<IRType>(target, Compare.Lhs);
        var rhsType = context.GetArgumentType<IRType>(target, Compare.Rhs);
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(lhsType) + CostUtility.GetMemoryAccess(rhsType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, CostUtility.GetCPUCyclesOfCompare()),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Compare target)
    {
        var outputType = context.GetReturnType<IRType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(outputType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType, 2),
        };
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Compare target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, Compare.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, Compare.Rhs);
        var operandTypes = TypeInference.BroadcastDistributeTypes(lhs, rhs);
        return (operandTypes[0], operandTypes[1]) switch
        {
            (TensorType a, TensorType b) => Visit(a, b),
            (DistributedType a, DistributedType b) => Visit(a, b),
            _ => new InvalidType($"{lhs} {rhs}"),
        };
    }

    public string Visit(IPrintOpContext context, Compare target)
    {
        if (context.Flags.HasFlag(PrinterFlags.Inline) || context.Flags.HasFlag(PrinterFlags.Script))
        {
            var op = target.CompareOp switch
            {
                CompareOp.Equal => "==",
                CompareOp.LowerOrEqual => "<=",
                CompareOp.GreaterOrEqual => ">=",
                CompareOp.GreaterThan => ">",
                CompareOp.LowerThan => "<",
                CompareOp.NotEqual => "!=",
                _ => throw new ArgumentOutOfRangeException(target.CompareOp.ToString()),
            };

            return $"({context.GetArgument(target, Compare.Lhs)} {op} {context.GetArgument(target, Compare.Rhs)})";
        }

        return context.GetDefault(target);
    }

    private bool Compute(CompareOp op, int a, int b) => op switch
    {
        CompareOp.Equal => a == b,
        CompareOp.LowerOrEqual => a <= b,
        CompareOp.GreaterOrEqual => a >= b,
        CompareOp.GreaterThan => a > b,
        CompareOp.LowerThan => a < b,
        CompareOp.NotEqual => a != b,
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private IRType Visit(TensorType lhs, TensorType rhs)
    {
        var broadcastType = TypeInference.BroadcastType(lhs, rhs);
        if (broadcastType is TensorType tensorType)
        {
            return tensorType with { DType = DataTypes.Boolean };
        }

        return broadcastType;
    }

    private IRType Visit(DistributedType a, DistributedType b)
    {
        if (a.Placement != b.Placement)
        {
            return new InvalidType("lhs rhs have different placement");
        }

        var rType = Visit(a.TensorType, b.TensorType);
        if (rType is not TensorType tensorType)
        {
            return rType;
        }

        return CheckSBP(tensorType, a, b);
    }
}
