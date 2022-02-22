// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Unary"/>.
/// </summary>
public class UnaryEvaluator : IEvaluator<Unary>, ITypeInferencer<Unary>, ICostEvaluator<Unary>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Unary unary)
    {
        var i = context.GetOrtArgumentValue(unary, Unary.Input);
        var result = unary.UnaryOp switch
        {
            UnaryOp.Abs => OrtKI.Abs(i),
            UnaryOp.Acos => OrtKI.Acos(i),
            UnaryOp.Acosh => OrtKI.Acosh(i),
            UnaryOp.Asin => OrtKI.Asin(i),
            UnaryOp.Asinh => OrtKI.Asinh(i),
            UnaryOp.Ceil => OrtKI.Ceil(i),
            UnaryOp.Cos => OrtKI.Cos(i),
            UnaryOp.Cosh => OrtKI.Cosh(i),
            UnaryOp.Exp => OrtKI.Exp(i),
            UnaryOp.Floor => OrtKI.Floor(i),
            UnaryOp.Log => OrtKI.Log(i),
            UnaryOp.Neg => OrtKI.Neg(i),
            UnaryOp.Round => OrtKI.Round(i),
            UnaryOp.Rsqrt => OrtKI.Rsqrt(i),
            UnaryOp.Sin => OrtKI.Sin(i),
            UnaryOp.Sinh => OrtKI.Sinh(i),
            UnaryOp.Sign => OrtKI.Sign(i),
            UnaryOp.Sqrt => OrtKI.Sqrt(i),
            UnaryOp.Square => OrtKI.Square(i),
            UnaryOp.Tanh => OrtKI.Tanh(i),
            UnaryOp.BitwiseNot => throw new NotSupportedException("NotSupported UnaryOp BitwiseNot"),
            UnaryOp.LogicalNot => OrtKI.Not(i),
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

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Unary target)
    {
        var returnType = context.GetReturnType<TensorType>();
        var arithm = returnType.Shape.Prod().FixedValue;
        return new(arithm, arithm * returnType.DType.SizeInBytes);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
