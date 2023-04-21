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

internal sealed class ShapeEvaluateProvider : IShapeEvaluateProvider
{
    private readonly IServiceProvider _serviceProvider;

    public ShapeEvaluateProvider(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public Expr EvaluateShapeExpr(Expr expr, IReadOnlyDictionary<Var, Expr[]>? varsMap = null)
    {
        if (expr.CheckedType is null)
        {
            expr.InferenceType();
        }

        if (expr.CheckedType is InvalidType)
        {
            throw new InvalidOperationException("Expr in Evaluator need a valid type");
        }

        if (expr.CheckedType is TensorType && expr.CheckedShape.IsFixed)
        {
            return expr.CheckedShape;
        }

        if (expr.CheckedType is TupleType tupleType)
        {
            bool fixedShape = tupleType.Fields.All(field => field is TensorType tensor && tensor.Shape.IsFixed);
            if (fixedShape)
            {
                var shapes = tupleType.Fields
                    .Select(field => (Expr)Tensor.From(((TensorType)field).Shape.ToValueArray()));
                return new IR.Tuple(shapes.ToArray());
            }
        }

        if (varsMap == null)
        {
            throw new InvalidOperationException();
        }

        var evaluatorVisitor = new ShapeEvaluateVisitor(varsMap);
        return evaluatorVisitor.Visit(expr);
    }

    public Expr EvaluateOpShapeExpr(Op op, IShapeEvaluateContext context)
    {
        var evaluatorType = typeof(IShapeEvaluator<>).MakeGenericType(op.GetType());
        var evaluator = (IShapeEvaluator)_serviceProvider.GetRequiredService(evaluatorType);
        return evaluator.Visit(context, op);
    }
}
