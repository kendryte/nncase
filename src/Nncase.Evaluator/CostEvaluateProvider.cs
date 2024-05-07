// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Nncase.CostModel;
using Nncase.IR;

namespace Nncase.Evaluator;

internal sealed class CostEvaluateProvider : ICostEvaluateProvider
{
    private readonly IServiceProvider _serviceProvider;

    public CostEvaluateProvider(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public Cost EvaluateCost(Expr expr, CompileOptions compileOptions)
    {
        if (expr.CheckedType is null)
        {
            expr.InferenceType();
        }

        if (expr.CheckedType is InvalidType)
        {
            throw new InvalidOperationException("Expr in Cost Evaluator need a valid type");
        }

        var evaluatorVisitor = new CostEvaluateVisitor(compileOptions);
        return evaluatorVisitor.Visit(expr);
    }

    public Cost EvaluateOpCost(Op op, ICostEvaluateContext context)
    {
        // TODO: Add inferencers cache.
        var evaluatorType = typeof(ICostEvaluator<>).MakeGenericType(op.GetType());
        var evaluator = (ICostEvaluator)_serviceProvider.GetRequiredService(evaluatorType);
        return evaluator.Visit(context, op);
    }
}
