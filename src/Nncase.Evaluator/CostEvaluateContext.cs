// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using Nncase.CostModel;
using Nncase.Evaluator;
using Nncase.IR;

namespace Nncase.Evaluator;

/// <summary>
/// Cost evaluator context.
/// </summary>
internal sealed class CostEvaluateContext : ICostEvaluateContext
{
    private readonly Dictionary<Expr, Cost> _exprMemo;
    private Call? _currentCall;

    public CostEvaluateContext(Dictionary<Expr, Cost> exprMemo)
    {
        _exprMemo = exprMemo;
    }

    public Call CurrentCall
    {
        get => _currentCall ?? throw new InvalidOperationException("Current call is not set in evaluator.");
        set => _currentCall = value;
    }

    public Cost GetArgumentCost(Op op, ParameterInfo parameter)
    {
        return GetCost(GetArgumentExpr(op, parameter));
    }

    public T GetArgumentType<T>(Op op, ParameterInfo parameter)
        where T : IRType
    {
        return (T)GetArgumentExpr(op, parameter).CheckedType;
    }

    public T GetReturnType<T>()
        where T : IRType
    {
        return (T)CurrentCall.CheckedType;
    }

    public bool TryGetConstArgument(Op op, ParameterInfo parameter, [MaybeNullWhen(false)] out Const @const)
    {
        @const = null;
        if (CurrentCall[parameter.Index] is Const c)
        {
            @const = c;
            return true;
        }

        return false;
    }

    private Expr GetArgumentExpr(Op op, ParameterInfo parameter)
    {
        if (op.GetType() == parameter.OwnerType)
        {
            return CurrentCall.Arguments[parameter.Index];
        }
        else
        {
            throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
        }
    }

    private Cost GetCost(Expr expr)
    {
        return _exprMemo[expr];
    }
}
