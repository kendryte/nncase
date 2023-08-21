﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.TIR;
using static Nncase.IR.F.Tensors;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Evaluator;

internal sealed class ShapeEvaluateContext : IShapeEvaluateContext
{
    private readonly Dictionary<Expr, Expr> _memo;

    public ShapeEvaluateContext(Dictionary<Expr, Expr> memo, ShapeExprCache cache)
    {
        _memo = memo;
        Cache = cache.Cache;
        VarMap = cache.VarMap;
    }

    public IReadOnlyDictionary<Var, Expr[]> VarMap { get; }

    public Call? CurrentCall { get; set; }

    // memo used by reference, can't make new _memo with memo.concat(cache)
    public Dictionary<Expr, Expr> Cache { get; set; }

    public Expr GetArgument(Op op, ParameterInfo parameter)
    {
        if (op.GetType() == parameter.OwnerType)
        {
            return GetCurrentCall().Arguments[parameter.Index];
        }
        else
        {
            throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
        }
    }

    public Expr[] GetArguments(Op op, params ParameterInfo[] paramsInfo)
    {
        return paramsInfo.Select(info => GetArgument(op, info)).ToArray();
    }

    public Expr GetArgumentShape(Op op, ParameterInfo parameter)
    {
        var expr = GetArgument(op, parameter);
        if (expr is Tuple tuple)
        {
            return new Tuple(tuple.Fields.ToArray().Select(v => Cast(GetResultFromMemo(v), DataTypes.Int32)).ToArray());
        }

        // call
        if (expr.CheckedType is TupleType)
        {
            var shape = expr.EvaluateShapeExpr(new ShapeExprCache(VarMap));
            if (shape is Call c && c.Target is IR.Math.Require && c.Arguments[IR.Math.Require.Value.Index] is Tuple tupleShapeExpr)
            {
                return new Tuple(tupleShapeExpr.Fields.ToArray().Select(expr => Cast(expr, DataTypes.Int32)).ToArray());
            }

            // for split
            else
            {
                // when it is if, it not tuple
                if (shape is If @if && @if.CheckedType is TupleType tupleType)
                {
                    return new Tuple(
                        Enumerable
                            .Range(0, tupleType.Fields.Count)
                            .Select(i => Cast(shape[i], DataTypes.Int32))
                            .ToArray());
                }
                else
                {
                    return new Tuple(((Tuple)shape).Fields.ToArray().Select(expr => Cast(expr, DataTypes.Int32)).ToArray());
                }
            }
        }

        var shapeExpr = GetResultFromMemo(expr);
        return Cast(shapeExpr, DataTypes.Int32);
    }

    public Expr GetArgumentRank(Op op, ParameterInfo parameter)
    {
        return StackScalar(Cast(GetArgumentShape(op, parameter)[0], DataTypes.Int32));
    }

    private Call GetCurrentCall() => CurrentCall ?? throw new InvalidOperationException("Current call is not set.");

    private Expr GetResultFromMemo(Expr expr)
    {
        if (_memo.ContainsKey(expr))
        {
            return _memo[expr];
        }

        throw new InvalidOperationException("Expr not found in memo and cache");
    }
}
