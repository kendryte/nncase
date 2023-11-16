// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Nncase.Diagnostics;
using Nncase.IR;

namespace Nncase.Evaluator;

internal sealed class EvaluateProvider : IEvaluateProvider
{
    private readonly IServiceProvider _serviceProvider;

    public EvaluateProvider(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public IValue Evaluate(Expr expr, IReadOnlyDictionary<Var, IValue>? varsValues = null, Dictionary<Type, IEvaluator>? evaluator_cache = null)
    {
        if (expr.CheckedType is null)
        {
            expr.InferenceType();
        }

        if (expr.CheckedType is InvalidType)
        {
            if (DumpScope.Current.IsEnabled(DumpFlags.Compile))
            {
                DumpScope.Current.DumpIR(expr, "EvaluateInvalid");
            }

            throw new InvalidOperationException("Expr in Evaluator need a valid type");
        }

        using var evaluatorVisitor = new EvaluateVisitor(varsValues ?? new Dictionary<Var, IValue>(), evaluator_cache ?? new());
        return evaluatorVisitor.Visit(expr);
    }

    public IValue EvaluateOp(Op op, IEvaluateContext context, Dictionary<Type, IEvaluator>? evaluator_cache = null)
    {
        var op_type = op.GetType();
        if (evaluator_cache is null)
        {
            var evaluatorType = typeof(IEvaluator<>).MakeGenericType(op_type);
            return ((IEvaluator)_serviceProvider.GetRequiredService(evaluatorType)).Visit(context, op);
        }

        if (!evaluator_cache.TryGetValue(op_type, out var evaluator))
        {
            var evaluatorType = typeof(IEvaluator<>).MakeGenericType(op_type);
            evaluator = (IEvaluator)_serviceProvider.GetRequiredService(evaluatorType);
            evaluator_cache.Add(op_type, evaluator);
        }

        return evaluator.Visit(context, op);
    }
}
