﻿// Copyright (c) Canaan Inc. All rights reserved.
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

    protected override Expr VisitLeafBaseFunction(BaseFunction expr)
    {
        return None.Default;
    }

    protected override Expr VisitLeafIf(If expr)
    {
        // todo: then or else??
        return Visit(expr.Then);
    }
    // protected override Expr DispatchVisit(Expr expr)
    // {
    //     // if (expr.Metadata.ShapeExpr is null)
    //     // {
    //         // expr.Metadata.ShapeExpr = base.DispatchVisit(expr);
    //     // }
    //
    //     return expr.Metadata.ShapeExpr;
    // }

    /// <inheritdoc/>
    protected override Expr VisitLeafConst(Const expr)
    {
        return expr.CheckedShape.ToValueArray();
    }

    /// <inheritdoc/>
    protected override Expr VisitLeafCall(Call expr)
    {
        _context.CurrentCall = expr;

        // function, VarMap merge expr
        return expr.Target switch
        {
            Op op => CompilerServices.EvaluateOpShapeExpr(op, _context),
            Function func => CompilerServices.EvaluateShapeExpr(func.Body, Merge(func.Parameters, expr.Arguments)),
            Fusion { ModuleKind: "stackvm" } func => CompilerServices.EvaluateShapeExpr(func.Body,
                Merge(func.Parameters, expr.Arguments)),
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

    private Dictionary<Var, Expr[]> Merge(ReadOnlySpan<Var> paramList, ReadOnlySpan<Expr> args)
    {
        var data = paramList.ToArray().Zip(args.ToArray().Select(arg =>
        {
            var result = Visit(arg);
            return Enumerable.Range(0, arg.CheckedShape.Rank).Select(i => result[i]).ToArray();
        })).Select(pair => new KeyValuePair<Var, Expr[]>(pair.First, pair.Second));
        var dict = _context.VarMap.Concat(data).ToDictionary(pair => pair.Key, pair => pair.Value);

        // Console.WriteLine("merge begin");
        // foreach (var (key, value) in dict)
        // {
        // Console.WriteLine(key.Name);
        // Console.WriteLine(value);
        // }
        // Console.WriteLine("merge end");
        return dict;
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

            // Console.WriteLine("in shape evaluator");
            // Console.WriteLine(expr.Name);
            // Console.WriteLine("VarMap");
            // foreach (var (key, value) in _context.VarMap)
            // {
            //     Console.WriteLine("Key:");
            //     Console.WriteLine(key.Name);
            //     Console.WriteLine(key.GlobalVarIndex);
            //     Console.WriteLine("Value:");
            //     foreach (var v in value)
            //     {
            //         if (v is Var var)
            //         {
            //             Console.WriteLine(var.Name);
            //             Console.WriteLine(var.GlobalVarIndex);
            //         }
            //         else
            //         {
            //             Console.WriteLine(v.ToString());
            //         }
            //
            //         Console.WriteLine(v.IsAlive);
            //     }
            // }
            //
            // if (!_context.VarMap.ContainsKey(expr))
            // {
            //     Console.WriteLine("error");
            //     Console.WriteLine(expr.Name);
            // }

            var shapeExpr = shape.Select((x, i) => x.IsFixed ? x.FixedValue : _context.VarMap[expr][i]).ToArray();

            // Console.WriteLine("ShapeExprList");
            // foreach (var expr1 in shapeExpr)
            // {
            //     Console.WriteLine("item");
            //     if (expr1 is Var v)
            //     {
            //         Console.WriteLine(v.Name);
            //         Console.WriteLine(v.GlobalVarIndex);
            //     }
            //     Console.WriteLine(expr1.IsAlive);
            //     Console.WriteLine(expr1);
            // }
            // Console.WriteLine("End");
            return IR.F.Tensors.Stack(new IR.Tuple(shapeExpr), 0);
        }

        throw new InvalidOperationException();
    }
}
