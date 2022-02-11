// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using CommonServiceLocator;
using Microsoft.Extensions.DependencyInjection;
using Nncase.IR;
using TorchSharp;

namespace Nncase.Evaluator;

internal sealed class EvaluateVisitor : ExprVisitor<Const, IRType>
{
    private readonly EvaluateContext _context;
    private readonly IReadOnlyDictionary<Var, Const> _varsValues;

    public EvaluateVisitor(IReadOnlyDictionary<Var, Const> varsValues)
    {
        _context = new EvaluateContext(ExpressionMemo, varsValues);
        _varsValues = varsValues;
    }

    public override Const VisitLeaf(Call expr)
    {
        _context.CurrentCall = expr;
        if (expr.Target is Function)
        {
            throw new NotImplementedException();
        }

        var target = (Op)expr.Target;
        return CompilerServices.EvaluateOp(target, _context);
    }

    public override Const VisitLeaf(Const expr)
    {
        return expr;
    }

    public override Const VisitLeaf(Op expr)
    {
        // todo:maybe a problem
        return Const.FromScalar(0);
    }

    public override Const VisitLeaf(Function expr)
    {
        return Const.FromScalar(0);
    }

    public override Const VisitLeaf(IR.Tuple expr)
    {
        return Const.FromScalar(0);
    }

    public override Const VisitLeaf(Var expr)
    {
        if (!_varsValues.TryGetValue(expr, out var result))
        {
            throw new InvalidProgramException($"Must Set Input For Var {expr.Name}!");
        }

        if (result is null)
        {
            throw new InvalidProgramException($"Must Set Input For Var {expr.Name}!");
        }

        if (result.ValueType != expr.CheckedType)
        {
            throw new InvalidProgramException(
              $"The Var {expr.Name} Require {expr.CheckedType} But Give {result.CheckedType}");
        }

        return result;
    }

    // when torch return a scalar, scalar's shape is {0}
    private static torch.Tensor FixShape(Expr expr, torch.Tensor tensor) =>
        expr.CheckedShape.IsScalar ? tensor.view(new long[] { }) : tensor;
}
