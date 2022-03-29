using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Numerics.Tensors;
using System.Reflection;
using Autofac;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Nncase.IR;

namespace Nncase.Evaluator;

internal sealed class EvaluateProvider : IEvaluateProvider
{
    private readonly IServiceProvider _serviceProvider;

    public EvaluateProvider(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public IValue Evaluate(Expr expr, IReadOnlyDictionary<Var, IValue>? varsValues = null)
    {
        if (expr.CheckedType is null)
        {
            expr.InferenceType();
        }

        if (expr.CheckedType is InvalidType)
        {
            throw new InvalidOperationException("Expr in Evaluator need a valid type");
        }

        var evaluatorVisitor = new EvaluateVisitor(varsValues ?? new Dictionary<Var, IValue>());
        return evaluatorVisitor.Visit(expr);
    }

    public IValue EvaluateOp(Op op, IEvaluateContext context)
    {
        // TODO: Add inferencers cache.
        var evaluatorType = typeof(IEvaluator<>).MakeGenericType(op.GetType());
        var evaluator = (IEvaluator)_serviceProvider.GetRequiredService(evaluatorType);
        return evaluator.Visit(context, op);
    }
}
