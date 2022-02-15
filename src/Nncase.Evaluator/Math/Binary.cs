// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Binary"/>.
/// </summary>
public partial class BinaryEvaluator : IEvaluator<Binary>, ITypeInferencer<Binary>, ICostEvaluator<Binary>
{
    /// <inheritdoc />
    public IValue Visit(IEvaluateContext context, Binary binary)
    {
        var a = context.GetOrtArgumentValue(binary, Binary.Lhs);
        var b = context.GetOrtArgumentValue(binary, Binary.Rhs);
        return (binary.BinaryOp switch
        {
            BinaryOp.Add => a + b,
            BinaryOp.Sub => a - b,
            BinaryOp.Mul => a * b,
            BinaryOp.Div => a / b,
            BinaryOp.Mod => a % b,
            BinaryOp.Min => OrtKI.Minimum(a, b),
            BinaryOp.Max => OrtKI.Maximum(a, b),
            BinaryOp.Pow => OrtKI.Pow(a, b),
            BinaryOp.BitwiseAnd => OrtKI.BitwiseAnd(a, b),
            BinaryOp.BitwiseOr => OrtKI.BitwiseOr(a, b),
            BinaryOp.BitwiseXor => OrtKI.BitwiseXor(a, b),
            BinaryOp.LogicalAnd => OrtKI.LogicalAnd(a, b),
            BinaryOp.LogicalOr => OrtKI.LogicalOr(a, b),
            BinaryOp.LogicalXor => OrtKI.LogicalXor(a, b),
            _ => throw new ArgumentOutOfRangeException(nameof(binary.BinaryOp)),
        }).ToType(context.CurrentCall.CheckedDataType.ToOrtType()).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Binary target)
    {
        var lhs = context.CheckArgumentType<TensorType>(target, Binary.Lhs);
        var rhs = context.CheckArgumentType<TensorType>(target, Binary.Rhs);
        return Visit(lhs, rhs);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Binary target)
    {
        var returnType = context.GetReturnType<TensorType>();
        var arithm = returnType.Shape.Prod().FixedValue;
        return new(arithm, arithm * returnType.DType.SizeInBytes);
    }

    private IRType Visit(TensorType lhs, TensorType rhs)
    {
        return TypeInference.BroadcastType(lhs, rhs);
    }
}
