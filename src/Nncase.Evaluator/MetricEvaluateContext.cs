// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using Nncase.Evaluator;
using Nncase.IR;

namespace Nncase.Evaluator;

/// <summary>
/// Metric evaluator context.
/// </summary>
internal sealed class MetricEvaluateContext : IMetricEvaluateContext
{
    private readonly Dictionary<Expr, Metric> _exprMemo;
    private Call? _currentCall;

    public MetricEvaluateContext(Dictionary<Expr, Metric> exprMemo)
    {
        _exprMemo = exprMemo;
    }

    public Call CurrentCall
    {
        get => _currentCall ?? throw new InvalidOperationException("Current call is not set in evaluator.");
        set => _currentCall = value;
    }

    public T GetArgument<T>(Op op, ParameterInfo parameter)
      where T : BaseFunction
    {
        return (T)CurrentCall[parameter];
    }

    public Metric GetArgumentMetric(Op op, ParameterInfo parameter)
    {
        return GetMetric(GetArgumentExpr(op, parameter));
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

    private Metric GetMetric(Expr expr)
    {
        return _exprMemo[expr];
    }
}
