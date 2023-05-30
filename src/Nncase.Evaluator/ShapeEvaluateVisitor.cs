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

    protected override Expr VisitLeafBaseFunction(BaseFunction expr)
    {
        return None.Default;
    }

    protected override Expr VisitLeafIf(If expr)
    {
        return new If(expr.Condition, Visit(expr.Then), Visit(expr.Else));
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

        // function, VarMap merge expr
        return expr.Target switch
        {
            Op op => CompilerServices.EvaluateOpShapeExpr(op, _context),
            Function func => CompilerServices.EvaluateShapeExpr(func.Body, MergeArgsVarMap(func.Parameters, expr.Arguments)),
            Fusion { ModuleKind: "stackvm" } func => CompilerServices.EvaluateShapeExpr(
                func.Body,
                MergeArgsVarMap(func.Parameters, expr.Arguments)),
            _ => throw new NotImplementedException(expr.Target.ToString()),
        };
    }

    /// <inheritdoc/>
    protected override Expr VisitLeafOp(Op expr)
    {
        return None.Default;
    }

    protected override Expr VisitLeafMarker(Marker marker)
    {
        return Visit(marker.Target);
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

            // PrintNotExistVarMap(expr);
            var shapeExpr = shape.Select((x, i) => x.IsFixed ? x.FixedValue : _context.VarMap[expr][i]).ToArray();
            return IR.F.Tensors.Stack(new IR.Tuple(shapeExpr), 0);
        }

        throw new InvalidOperationException();
    }

    private Dictionary<Var, Expr[]> MergeArgsVarMap(ReadOnlySpan<Var> paramList, ReadOnlySpan<Expr> args)
    {
        var data = paramList.ToArray().Zip(args.ToArray().Select(arg =>
        {
            var result = Visit(arg);
            return Enumerable.Range(0, arg.CheckedShape.Rank).Select(i => result[i]).ToArray();
        })).Select(pair => new KeyValuePair<Var, Expr[]>(pair.First, pair.Second));
        var dict = _context.VarMap.Concat(data).ToDictionary(pair => pair.Key, pair => pair.Value);
        return dict;
    }

    private void PrintNotExistVarMap(Var expr)
    {
        if (!_context.VarMap.ContainsKey(expr))
        {
            Console.WriteLine("key not found error");
            Console.WriteLine(expr.Name);
            Console.WriteLine("VarMap");
            foreach (var (key, value) in _context.VarMap)
            {
                Console.WriteLine("Key:");
                Console.WriteLine(key.Name);
                Console.WriteLine(key.GlobalVarIndex);
                Console.WriteLine("Value:");
                foreach (var v in value)
                {
                    if (v is Var var)
                    {
                        Console.WriteLine(var.Name);
                        Console.WriteLine(var.GlobalVarIndex);
                    }
                    else
                    {
                        Console.WriteLine(v.ToString());
                    }

                    Console.WriteLine(v.IsAlive);
                }
            }
        }
    }
}
