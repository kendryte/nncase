// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Compare"/>.
/// </summary>
public class CompareEvaluator : IEvaluator<Compare>, ITypeInferencer<Compare>, ICostEvaluator<Compare>, IOpPrinter<Compare>, IShapeEvaluator<Compare>
{
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
        var lhsType = context.GetArgumentType<TensorType>(target, Compare.Lhs);
        var rhsType = context.GetArgumentType<TensorType>(target, Compare.Rhs);
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(lhsType) + CostUtility.GetMemoryAccess(rhsType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, CostUtility.GetCPUCyclesOfCompare()),
        };
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Compare target)
    {
        var lhs = context.CheckArgumentType<TensorType>(target, Compare.Lhs);
        var rhs = context.CheckArgumentType<TensorType>(target, Compare.Rhs);
        return Visit(lhs, rhs);
    }

    public string Visit(IIRPrinterContext context, Compare target, bool iLmode)
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
        return $"{context.GetArgument(target, Compare.Lhs)} {op} {context.GetArgument(target, Compare.Rhs)}";
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
        return ((TensorType)TypeInference.BroadcastType(lhs, rhs)) with { DType = DataTypes.Boolean };
    }

    public Expr Visit(IShapeEvaluateContext context, Compare target)
    {
        var lhs = context.GetArgumentShape(target, Compare.Lhs);
        var rhs = context.GetArgumentShape(target, Compare.Rhs);
        return ShapeExprUtility.BroadcastShape(lhs, rhs);
    }
}
