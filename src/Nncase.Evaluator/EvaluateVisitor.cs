// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive;
using DryIoc.ImTools;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Evaluator;

internal sealed class EvaluateVisitor : ExprVisitor<IValue, Unit>, IDisposable
{
    private readonly EvaluateContext _context;
    private readonly IReadOnlyDictionary<IVar, IValue> _varsValues;
    private readonly EvaluatorDumpManager _dumpManager;
    private readonly Dictionary<Type, IEvaluator> _evaluator_cache;

    public EvaluateVisitor(IReadOnlyDictionary<IVar, IValue> varsValues, Dictionary<Type, IEvaluator> evaluator_cache)
    {
        _context = new EvaluateContext(this, ExprMemo);
        _evaluator_cache = evaluator_cache;
        _varsValues = varsValues;
        _dumpManager = new EvaluatorDumpManager(DumpScope.Current.CreateSubDummper("Evaluate", null), expr => _context.GetValue(expr).AsTensors());
        _dumpManager.RegisterDumpCallbacks(RegisterBeforeCallback, RegisterAfterCallback);
    }

    private event Action<Expr>? BeforeCallAction;

    private event Action<Expr>? AfterCallAction;

    public void Dispose()
    {
        _dumpManager.Dispose();
    }

    /// <inheritdoc/>
    protected override IValue VisitLeafBaseFunction(BaseFunction expr) => NoneValue.Default;

    /// <inheritdoc/>
    protected override IValue VisitLeafConst(Const expr) => Value.FromConst(expr);

    /// <inheritdoc/>
    protected override IValue VisitLeafMarker(Marker expr) => ExprMemo[expr.Target];

    /// <inheritdoc/>
    protected override IValue VisitLeafNone(None expr) => NoneValue.Default;

    /// <inheritdoc/>
    protected override IValue VisitLeafTuple(IR.Tuple expr)
    {
        var fields = expr.Fields.AsValueEnumerable().Select(x => ExprMemo[x]);
        var value = new TupleValue(fields.ToArray());
        return value;
    }

    /// <inheritdoc/>
    protected override IValue VisitLeafVar(Var expr)
    {
        if (!_varsValues.TryGetValue(expr, out var value))
        {
            throw new ArgumentException($"Must Set Input For Var {expr.Name}!");
        }

        if (expr.CheckedType is not AnyType)
        {
            if (value.Type is TensorType resultType)
            {
                if (expr.CheckedShape.IsRanked)
                {
                    if (expr.CheckedDataType != resultType.DType)
                    {
                        throw new ArgumentException($"DataType mismatch. The Var {expr.Name} Require {expr.CheckedDataType} But Give {resultType.DType}");
                    }

                    if (!expr.CheckedShape.IsAssignableFrom(resultType.Shape))
                    {
                        throw new ArgumentException(
                            $"Shape mismatch. The Var {expr.Name} Require {expr.CheckedShape} But Give {resultType.Shape}");
                    }
                }
            }
        }

        return value;
    }

    /// <inheritdoc/>
    protected override IValue VisitCall(Call expr)
    {
        if (HasVisited(expr, out var result))
        {
            return result;
        }

        foreach (var param in expr.Arguments)
        {
            Visit(param);
        }

        _context.CurrentCall = expr;
        BeforeCallAction?.Invoke(expr);
        var value = MarkVisited(expr, VisitLeafCall(expr));
        AfterCallAction?.Invoke(expr);
        return value;
    }

    protected override IValue VisitLeafCall(Call expr) => EvaluateCallable(expr.Target, expr.Arguments);

    protected override IValue VisitIf(If expr)
    {
        if (HasVisited(expr, out var result))
        {
            return result;
        }

        Visit(expr.Condition);

        foreach (var param in expr.Arguments)
        {
            Visit(param);
        }

        _context.CurrentCall = expr;
        BeforeCallAction?.Invoke(expr);
        var value = MarkVisited(expr, VisitLeafIf(expr));
        AfterCallAction?.Invoke(expr);
        return value;
    }

    protected override IValue VisitLeafIf(If expr)
    {
        bool cond = Visit(expr.Condition).AsTensor().ToScalar<bool>();
        return cond ? EvaluateCallable(expr.Then, expr.Arguments) : EvaluateCallable(expr.Else, expr.Arguments);
    }

    protected override IValue VisitLeafShape(Shape expr)
    {
        var dims = expr.Dimensions.AsValueEnumerable().Select(x => ExprMemo[x].AsTensor().ToScalar<long>()).ToArray();
        return new ShapeValue(dims);
    }

    private IValue EvaluateCallable(Expr callable, ReadOnlySpan<Expr> arguments)
    {
        return callable switch
        {
            Op op => CompilerServices.EvaluateOp(op, _context, _evaluator_cache),
            Function func => CompilerServices.Evaluate(func.Body, CreateFunctionEvaluateArguments(func.Parameters, arguments), _evaluator_cache),
            Fusion fusion => CompilerServices.Evaluate(fusion.Body, CreateFunctionEvaluateArguments(fusion.Parameters, arguments), _evaluator_cache),
            _ => throw new NotImplementedException(callable.ToString()),
        };
    }

    private IReadOnlyDictionary<IVar, IValue> CreateFunctionEvaluateArguments(ReadOnlySpan<IVar> parameters, ReadOnlySpan<Expr> arguments)
    {
        var values = new Dictionary<IVar, IValue>(_varsValues);
        for (int i = 0; i < parameters.Length; i++)
        {
            values.Add(parameters[i], ExprMemo[arguments[i]]);
        }

        return values;
    }

    private void RegisterBeforeCallback(string name, Action<Expr> action)
    {
        BeforeCallAction += action;
    }

    private void RegisterAfterCallback(string name, Action<Expr> action)
    {
        AfterCallAction += action;
    }
}
