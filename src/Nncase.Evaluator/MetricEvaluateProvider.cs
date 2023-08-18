// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Nncase.IR;

namespace Nncase.Evaluator;

internal sealed class MetricEvaluateProvider : IMetricEvaluateProvider
{
    private readonly IServiceProvider _serviceProvider;

    public MetricEvaluateProvider(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public Dictionary<Expr, Metric> EvaluateMetric(Expr expr)
    {
        if (expr.CheckedType is null)
        {
            expr.InferenceType();
        }

        if (expr.CheckedType is InvalidType)
        {
            throw new InvalidOperationException("Expr in Metric Evaluator need a valid type");
        }

        var evaluatorVisitor = new MetricEvaluateVisitor();
        evaluatorVisitor.Visit(expr);

        return evaluatorVisitor.ExprMemo;
    }

    public Metric EvaluateOpMetric(Op op, IMetricEvaluateContext context)
    {
        // TODO: Add inferencers cache.
        var evaluatorType = typeof(IMetricEvaluator<>).MakeGenericType(op.GetType());
        var evaluator = (IMetricEvaluator)_serviceProvider.GetRequiredService(evaluatorType);
        return evaluator.Visit(context, op);
    }
}
