// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;
using TorchSharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Unary"/>.
/// </summary>
public class UnaryEvaluator : IEvaluator<Unary>, ITypeInferencer<Unary>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Unary unary)
    {
        var i = context.GetTorchArgumentValue(unary, Unary.Input);
        var result = unary.UnaryOp switch
        {
            UnaryOp.Abs => torch.abs(i),
            UnaryOp.Acos => torch.acos(i),
            UnaryOp.Acosh => torch.acosh(i),
            UnaryOp.Asin => torch.asin(i),
            UnaryOp.Asinh => torch.asinh(i),
            UnaryOp.Ceil => torch.ceil(i),
            UnaryOp.Cos => torch.cos(i),
            UnaryOp.Cosh => torch.cosh(i),
            UnaryOp.Exp => torch.exp(i),
            UnaryOp.Floor => torch.floor(i),
            UnaryOp.Log => torch.log(i),
            UnaryOp.Neg => torch.neg(i),
            UnaryOp.Round => torch.round(i),
            UnaryOp.Rsqrt => torch.rsqrt(i),
            UnaryOp.Sin => torch.sin(i),
            UnaryOp.Sinh => torch.sinh(i),
            UnaryOp.Sign => torch.sign(i),
            UnaryOp.Sqrt => torch.sqrt(i),
            UnaryOp.Square => torch.square(i),
            UnaryOp.Tanh => torch.tanh(i),
            UnaryOp.BitwiseNot => torch.bitwise_not(i),
            UnaryOp.LogicalNot => torch.logical_not(i),
            _ => throw new ArgumentOutOfRangeException(nameof(unary.UnaryOp)),
        };
        return result.ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Unary target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Unary.Input);
        return Visit(input);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
