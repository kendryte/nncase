// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Compare"/>.
/// </summary>
public class CompareEvaluator : IEvaluator<Compare>, ITypeInferencer<Compare>, IOpPrinter<Compare>
{
    /// <inheritdoc />
    public IValue Visit(IEvaluateContext context, Compare target)
    {
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
            _ => throw new ArgumentOutOfRangeException(target.CompareOp.ToString())
        };
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Compare target)
    {
        var lhs = context.CheckArgumentType<TensorType>(target, Compare.Lhs);
        var rhs = context.CheckArgumentType<TensorType>(target, Compare.Rhs);
        return Visit(lhs, rhs);
    }

    private IRType Visit(TensorType lhs, TensorType rhs)
    {
        return ((TensorType)TypeInference.BroadcastType(lhs, rhs)) with { DType = DataTypes.Boolean };
    }

    public string Visit(IIRPrinterContext context, Compare target, bool ILmode)
    {
        var op = target.CompareOp switch
        {
            CompareOp.Equal => "==",
            CompareOp.LowerOrEqual => "<=",
            CompareOp.GreaterOrEqual => ">=",
            CompareOp.GreaterThan => ">",
            CompareOp.LowerThan => "<",
            CompareOp.NotEqual => "!=",
            _ => throw new ArgumentOutOfRangeException(target.CompareOp.ToString())
        };
        return $"{context.GetArgument(target, Compare.Lhs)} {op} {context.GetArgument(target, Compare.Rhs)}";
    }
}
