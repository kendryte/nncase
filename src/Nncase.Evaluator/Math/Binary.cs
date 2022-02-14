// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;
using TorchSharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Binary"/>.
/// </summary>
public partial class BinaryEvaluator : IEvaluator<Binary>, ITypeInferencer<Binary>
{
    /// <inheritdoc />
    public IValue Visit(IEvaluateContext context, Binary binary)
    {
        var a = context.GetTorchArgumentValue(binary, Binary.Lhs);
        var b = context.GetTorchArgumentValue(binary, Binary.Rhs);
        return (binary.BinaryOp switch
        {
            BinaryOp.Add => a + b,
            BinaryOp.Sub => a - b,
            BinaryOp.Mul => a * b,
            BinaryOp.Div => a / b,
            BinaryOp.Mod => a % b,
            BinaryOp.Min => torch.minimum(a, b),
            BinaryOp.Max => torch.maximum(a, b),
            BinaryOp.Pow => torch.pow(a, b),
            BinaryOp.BitwiseAnd => torch.bitwise_and(a, b),
            BinaryOp.BitwiseOr => torch.bitwise_or(a, b),
            BinaryOp.BitwiseXor => torch.bitwise_xor(a, b),
            BinaryOp.LogicalAnd => torch.logical_and(a, b),
            BinaryOp.LogicalOr => torch.logical_or(a, b),
            BinaryOp.LogicalXor => torch.logical_xor(a, b),
            _ => throw new ArgumentOutOfRangeException(nameof(binary.BinaryOp)),
        }).to_type(context.CurrentCall.CheckedDataType.ToTorchType()).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Binary target)
    {
        var lhs = context.CheckArgumentType<TensorType>(target, Binary.Lhs);
        var rhs = context.CheckArgumentType<TensorType>(target, Binary.Rhs);
        return Visit(lhs, rhs);
    }

    private IRType Visit(TensorType lhs, TensorType rhs)
    {
        return TypeInference.BroadcastType(lhs, rhs);
    }
}
