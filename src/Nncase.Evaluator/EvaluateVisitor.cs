// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using CommonServiceLocator;
using Microsoft.Extensions.DependencyInjection;
using Nncase.IR;

namespace Nncase.Evaluator;

internal sealed class EvaluateVisitor : ExprVisitor<IValue, IRType>
{
    private readonly EvaluateContext _context;
    private readonly IReadOnlyDictionary<Var, IValue> _varsValues;

    public EvaluateVisitor(IReadOnlyDictionary<Var, IValue> varsValues)
    {
        _context = new EvaluateContext(ExpressionMemo);
        _varsValues = varsValues;
    }

    public override IValue VisitLeaf(Call expr)
    {
        _context.CurrentCall = expr;
        return expr.Target switch
        {
            Op op => CompilerServices.EvaluateOp(op, _context),
            Function func => CompilerServices.Evaluate(func.Body, func.Parameters.Zip(expr.Parameters).ToDictionary(kv => kv.First, kv => Visit(kv.Second), (IEqualityComparer<Var>)ReferenceEqualityComparer.Instance)),
            _ => throw new NotImplementedException(expr.Target.ToString())
        };
    }

    public override IValue VisitLeaf(Const expr)
    {
        return Value.FromConst(expr);
    }

    public override IValue VisitLeaf(None expr)
    {
        return Value.None;
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

        if (expr.CheckedType is not AnyType && result.Type != expr.CheckedType)
        {
            throw new ArgumentException(
              $"The Var {expr.Name} Require {expr.CheckedType} But Give {result.Type}");
        }

        return result;
    }
}
