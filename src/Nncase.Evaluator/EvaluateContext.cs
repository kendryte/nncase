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
    private readonly Dictionary<Expr, Const> _exprMemo;
    private readonly IReadOnlyDictionary<Var, Const> _varsValues;
    private Call? _currentCall;

    public EvaluateContext(Dictionary<Expr, Const> exprMemo, IReadOnlyDictionary<Var, Const> varsValues)
    {
        _exprMemo = exprMemo;
        _varsValues = varsValues;
    }

    public Call CurrentCall
    {
        get => _currentCall ?? throw new InvalidOperationException("Current call is not set in evaluator.");
        set => _currentCall = value;
    }

    public Const GetValue(Expr expr)
    {
        return _exprMemo[expr];
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

    public T GetArgumentValueAsScalar<T>(Op op, ParameterInfo parameter)
        where T : unmanaged
    {
        return GetArgumentValue(op, parameter).ToScalar<T>();
    }

    public T[] GetArgumentValueAsArray<T>(Op op, ParameterInfo parameter)
        where T : unmanaged
    {
        return GetArgumentValue(op, parameter).ToArray<T>();
    }

    public Const GetArgumentValue(Op op, ParameterInfo parameter)
    {
        var expr = GetArgumentExpr(op, parameter);
        if (expr is Const constValue)
        {
            return constValue;
        }
        else
        {
            // maybe a valid type but not const
            return GetValue(expr);
        }
    }

    public TensorType GetTensorType(Expr expr)
    {
        var resultType = expr.CheckedType ?? throw new InvalidOperationException($"Expr {expr} don't have CheckedType.");
        return resultType is TensorType resultTensorType ?
            resultTensorType :
            throw new InvalidOperationException($"Expr {expr} is not a TensorType.");
    }

    public TensorType CurrentCallResultTensorType() => GetTensorType(CurrentCall);
}

/// <summary>
/// Evaluate context extensions.
/// </summary>
public static class EvaluateContextExtensions
{
    public static torch.Tensor GetTorchArgumentValue(this IEvaluateContext context, Op op, ParameterInfo parameter)
    {
        return context.GetArgumentValue(op, parameter).ToTorchTensor();
    }

    public static torch.Tensor GetTorchValue(this IEvaluateContext context, Expr expr)
    {
        return context.GetValue(expr).ToTorchTensor();
    }

    public static Tensorflow.Tensor GetTFArgumentValue(this IEvaluateContext context, Op op, ParameterInfo parameter)
    {
        return context.GetArgumentValue(op, parameter).ToTFTensor();
    }
}
