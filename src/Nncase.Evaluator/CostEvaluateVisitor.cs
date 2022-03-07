// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using TorchSharp;

namespace Nncase.Evaluator;

internal sealed class CostEvaluateVisitor : ExprVisitor<Cost, IRType>
{
    private readonly CostEvaluateContext _context;
    private readonly IReadOnlyDictionary<Var, Cost> _varsValues;

    public CostEvaluateVisitor(IReadOnlyDictionary<Var, Cost> varsValues)
    {
        _context = new CostEvaluateContext(ExpressionMemo);
        _varsValues = varsValues;
    }

    public override Cost VisitLeaf(Call expr)
    {
        _context.CurrentCall = expr;
        if (expr.Target is Function)
        {
            throw new NotImplementedException();
        }

        var target = (Op)expr.Target;
        return CompilerServices.EvaluateOpCost(target, _context);
    }

    public override Cost VisitLeaf(Const expr)
    {
        return expr switch
        {
            TensorConst tc => new(0, tc.Value.BytesBuffer.Length),
            TupleConst tc => tc.Fields.Select(VisitLeaf).Sum(),
            _ => throw new ArgumentException("Invalid const type."),
        };
    }

    public override Cost VisitLeaf(Op expr)
    {
        return Cost.Zero;
    }

    public override Cost VisitLeaf(Function expr)
    {
        return Cost.Zero;
    }

    public override Cost VisitLeaf(IR.Tuple expr)
    {
        return expr.Fields.Select(Visit).Sum();
    }

    public override Cost VisitLeaf(Var expr)
    {
        if (!_varsValues.TryGetValue(expr, out var result))
        {
            result = Cost.Zero;
        }

        return result;
    }
}
