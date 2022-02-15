// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using Nncase.Evaluator;
using Nncase.IR;
using TorchSharp;

namespace Nncase.Evaluator;

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

    public Expr GetArgumentExpr(Op op, ParameterInfo parameter)
    {
        if (op.GetType() == parameter.OwnerType)
        {
            return CurrentCall.Parameters[parameter.Index];
        }
        else
        {
            throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
        }
    }

    public IValue GetValue(Expr expr)
    {
        return _exprMemo[expr];
    }
}

/// <summary>
/// Evaluate context extensions.
/// </summary>
public static class EvaluateContextExtensions
{
    public static torch.Tensor GetTorchArgumentValue(this IEvaluateContext context, Op op, ParameterInfo parameter)
    {
        return context.GetArgumentValue(op, parameter).AsTensor().ToTorchTensor();
    }

    public static torch.Tensor GetTorchValue(this IEvaluateContext context, Expr expr)
    {
        return context.GetValue(expr).AsTensor().ToTorchTensor();
    }

    public static Tensorflow.Tensor GetTFArgumentValue(this IEvaluateContext context, Op op, ParameterInfo parameter)
    {
        return context.GetArgumentValue(op, parameter).AsTensor().ToTFTensor();
    }

    public static OrtKISharp.Tensor GetOrtArgumentValue(this IEvaluateContext context, Op op, ParameterInfo parameter)
    {
        return context.GetArgumentValue(op, parameter).AsTensor().ToOrtTensor();
    }
    
    public static OrtKISharp.Tensor GetOrtValue(this IEvaluateContext context, Expr expr)
    {
        return context.GetValue(expr).AsTensor().ToOrtTensor();
    }
}
