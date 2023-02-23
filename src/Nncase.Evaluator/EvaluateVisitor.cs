// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Extensions.DependencyInjection;
using Nncase.Diagnostics;
using Nncase.IR;

namespace Nncase.Evaluator;

internal sealed class EvaluateVisitor : ExprVisitor<IValue, IRType>, IDisposable
{
    private readonly EvaluateContext _context;
    private readonly IReadOnlyDictionary<Var, IValue> _varsValues;
    private readonly EvaluatorDumpManager _dumpManager;
    private readonly Dictionary<Type, IEvaluator> _evaluator_cache;

    public EvaluateVisitor(IReadOnlyDictionary<Var, IValue> varsValues, Dictionary<Type, IEvaluator> evaluator_cache)
    {
        _context = new EvaluateContext(ExpressionMemo);
        _evaluator_cache = evaluator_cache;
        _varsValues = varsValues;
        _dumpManager = new EvaluatorDumpManager(DumpScope.Current.CreateSubDummper("Evaluate", null), expr => _context.GetValue(expr).AsTensors());
        _dumpManager.RegisterDumpCallbacks(RegisterBeforeCallback, RegisterAfterCallback);
    }

    /// <inheritdoc/>
    public override IValue VisitLeaf(Call expr)
    {
        _context.CurrentCall = expr;
        return expr.Target switch
        {
            Op op => CompilerServices.EvaluateOp(op, _context, _evaluator_cache),
            Function func => CompilerServices.Evaluate(func.Body, func.Parameters.Zip(expr.Parameters).ToDictionary(kv => kv.First, kv => Visit(kv.Second), (IEqualityComparer<Var>)ReferenceEqualityComparer.Instance), _evaluator_cache),
            Fusion { ModuleKind: "stackvm" } fusion => CompilerServices.Evaluate(fusion.Body, fusion.Parameters.Zip(expr.Parameters).ToDictionary(kv => kv.First, kv => Visit(kv.Second), (IEqualityComparer<Var>)ReferenceEqualityComparer.Instance), _evaluator_cache),
            _ => throw new NotImplementedException(expr.Target.ToString()),
        };
    }

    public override IValue VisitLeaf(If @if)
    {
        bool cond = @if.Condition.Evaluate(_varsValues, _evaluator_cache).AsTensor().ToScalar<bool>();
        return (cond ? @if.Then : @if.Else).Evaluate(_varsValues, _evaluator_cache);
    }

    public override IValue VisitLeaf(Const expr)
    {
        return Value.FromConst(expr);
    }

    public override IValue VisitLeaf(None expr)
    {
        return Value.None;
    }

    public override IValue VisitLeaf(Marker expr)
    {
        return Visit(expr.Target);
    }

    public override IValue VisitLeaf(Op expr)
    {
        // Value of Op is not needed in evaluate context.
        return null!;
    }

    public override IValue Visit(Function expr)
    {
        // Value of Function is not needed in evaluate context.
        return null!;
    }

    public override IValue Visit(Fusion expr)
    {
        return null!;
    }

    public override IValue VisitLeaf(IR.Tuple expr)
    {
        var fields = expr.Fields.Select(x => Visit(x));
        return new TupleValue(fields.ToArray());
    }

    public override IValue VisitLeaf(Var expr)
    {
        if (!_varsValues.TryGetValue(expr, out var result))
        {
            throw new ArgumentException($"Must Set Input For Var {expr.Name}!");
        }

        if (result is null)
        {
            throw new ArgumentException($"Must Set Input For Var {expr.Name}!");
        }

        if (expr.CheckedType is not AnyType)
        {
            if (result.Type is TensorType resultType)
            {
                if (expr.CheckedShape.IsUnranked)
                {
                    return result;
                }

                if (expr.CheckedDataType != resultType.DType)
                {
                    throw new ArgumentException($"DataType mismatch. The Var {expr.Name} Require {expr.CheckedDataType} But Give {resultType.DType}");
                }

                var s = expr.CheckedShape.Zip(resultType.Shape).ToArray();
                var matchedShape = s.Aggregate(true, (b, dims) => b && (dims.First.IsUnknown || dims.Second.IsUnknown || dims.First == dims.Second));
                if (!matchedShape)
                {
                    throw new ArgumentException(
                        $"Shape mismatch. The Var {expr.Name} Require {expr.CheckedShape} But Give {resultType.Shape}");
                }
            }
        }

        return result;
    }

    public void Dispose()
    {
        _dumpManager.Dispose();
    }
}
