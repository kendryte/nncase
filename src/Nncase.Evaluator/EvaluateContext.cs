// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using Nncase.Evaluator;
using Nncase.IR;

namespace Nncase.Evaluator;

/// <summary>
/// Evaluate context extensions.
/// </summary>
public static class EvaluateContextExtensions
{
    public static OrtKISharp.Tensor GetOrtArgumentValue(this IEvaluateContext context, Op op, ParameterInfo parameter)
    {
        return context.GetArgumentValue(op, parameter).AsTensor().ToOrtTensor();
    }

    public static OrtKISharp.Tensor GetOptionalOrtArgumentValue(this IEvaluateContext context, Op op, ParameterInfo parameter, Tensor tensor)
    {
        return context.GetOptionalArgumentValue(op, parameter).Match(
            x => x.AsTensor(),
            () => tensor).ToOrtTensor();
    }

    public static OrtKISharp.Tensor GetInt64OrtTensorArgumentValue(this IEvaluateContext context, Op op, ParameterInfo parameter)
    {
        var tensor = context.GetArgumentValue(op, parameter).AsTensor().Cast<long>();
        return tensor.Shape.IsScalar ? tensor.ScalarToOrtTensor() : tensor.ToOrtTensor();
    }

    public static Tensorflow.Tensor GetTFArgumentValue(this IEvaluateContext context, Op op, ParameterInfo parameter)
    {
        return context.GetArgumentValue(op, parameter).AsTensor().ToTFTensor();
    }
}

/// <summary>
/// Evaluator context.
/// </summary>
internal sealed class EvaluateContext : IEvaluateContext
{
    private readonly Dictionary<Expr, IValue> _exprMemo;
    private Call? _currentCall;

    public EvaluateContext(Dictionary<Expr, IValue> exprMemo)
    {
        _exprMemo = exprMemo;
    }

    public Call CurrentCall
    {
        get => _currentCall ?? throw new InvalidOperationException("Current call is not set in evaluator.");
        set => _currentCall = value;
    }

    public IValue GetValue(Expr expr) => _exprMemo[expr];

    public IValue GetArgumentValue(Op op, ParameterInfo parameter)
    {
        var expr = op.GetType() == parameter.OwnerType
            ? CurrentCall.Arguments[parameter.Index]
            : throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
        return _exprMemo[expr];
    }
}
