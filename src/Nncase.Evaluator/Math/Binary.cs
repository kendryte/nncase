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
public partial class BinaryEvaluator : IEvaluator<Binary>, ITypeInferencer<Binary>, ICostEvaluator<Binary>, IOpPrinter<Binary>
{
    /// <inheritdoc />
    public IValue Visit(IEvaluateContext context, Binary binary)
    {
        var lhs = context.GetArgumentValue(binary, Binary.Lhs);
        var a = lhs.AsTensor().ToOrtTensor();
        var b = context.GetOrtArgumentValue(binary, Binary.Rhs);
        return (binary.BinaryOp switch
        {
            BinaryOp.Add => a + b,
            BinaryOp.Sub => a - b,
            BinaryOp.Mul => a * b,
            BinaryOp.Div => a / b,
            BinaryOp.Mod => a % b,
            BinaryOp.Min => OrtKI.Min(new[] { a, b }),
            BinaryOp.Max => OrtKI.Max(new[] { a, b }),
            BinaryOp.Pow => OrtKI.Pow(a, b),
            BinaryOp.BitwiseAnd => throw new NotSupportedException("NotSupported BinaryOp BitwiseAnd"),
            BinaryOp.BitwiseOr => throw new NotSupportedException("NotSupported BinaryOp BitwiseOr"),
            BinaryOp.BitwiseXor => throw new NotSupportedException("NotSupported BinaryOp BitwiseXor"),
            BinaryOp.LogicalAnd => OrtKI.And(a, b),
            BinaryOp.LogicalOr => OrtKI.Or(a, b),
            BinaryOp.LogicalXor => OrtKI.Xor(a, b),
            BinaryOp.LeftShift => OrtKI.LeftShift(a, b),
            BinaryOp.RightShift => OrtKI.RightShift(a, b),
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

    /// <inheritdoc/>
    public string Visit(IIRPrinterContext context, Binary target, bool ILmode)
    {
        var lhs = context.GetArgument(target, Binary.Lhs);
        var rhs = context.GetArgument(target, Binary.Rhs);

        return $"({lhs.Serialize()} {context.Get(target).Serialize()} {rhs.Serialize()})";
    }

    private IRType Visit(TensorType lhs, TensorType rhs)
    {
        return TypeInference.BroadcastType(lhs, rhs);
    }
}
