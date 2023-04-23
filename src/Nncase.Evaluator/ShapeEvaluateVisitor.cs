// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using Nncase.IR;

namespace Nncase.Evaluator;

internal sealed class ShapeEvaluateVisitor : ExprVisitor<Expr, Unit>
{
    private readonly ShapeEvaluateContext _context;

    public ShapeEvaluateVisitor(IReadOnlyDictionary<Var, Expr[]> varMap)
    {
        _context = new ShapeEvaluateContext(ExprMemo, varMap);
    }

    protected override Expr DispatchVisit(Expr expr)
    {
        if (expr.Metadata.ShapeExpr is null)
        {
            expr.Metadata.ShapeExpr = base.DispatchVisit(expr);
        }

        return expr.Metadata.ShapeExpr;
    }

    /// <inheritdoc/>
    protected override Expr VisitLeafConst(Const expr)
    {
        return expr.CheckedShape.ToValueArray();
    }

    /// <inheritdoc/>
    protected override Expr VisitLeafCall(Call expr)
    {
        _context.CurrentCall = expr;

        return expr.Target switch
        {
            Op op => CompilerServices.EvaluateOpShapeExpr(op, _context),
            Function func => CompilerServices.EvaluateShapeExpr(func.Body),
            _ => throw new NotImplementedException(expr.Target.ToString()),
        };
    }

    /// <inheritdoc/>
    protected override Expr VisitLeafOp(Op expr)
    {
        return None.Default;
    }

    /// <inheritdoc/>
    protected override Expr VisitLeafTuple(IR.Tuple expr)
    {
        return new IR.Tuple(expr.Fields.ToArray().Select(Visit).ToArray());
    }

    /// <inheritdoc/>
    protected override Expr VisitLeafVar(Var expr)
    {
        if (expr.TypeAnnotation is TensorType tensorType)
        {
            var shape = tensorType.Shape;
            if (shape.IsFixed)
            {
                return shape.ToValueArray();
            }

            var shapeExpr = shape.Select((x, i) => x.IsFixed ? x.FixedValue : _context.VarMap[expr][i]).ToArray();
            return IR.F.Tensors.Stack(new IR.Tuple(shapeExpr), 0);
        }

        throw new InvalidOperationException();
    }
}
