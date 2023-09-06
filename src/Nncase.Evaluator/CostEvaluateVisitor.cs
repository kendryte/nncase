// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;

namespace Nncase.Evaluator;

internal sealed class CostEvaluateVisitor : ExprVisitor<Cost, Unit>
{
    private readonly CostEvaluateContext _context;

    public CostEvaluateVisitor()
    {
        _context = new CostEvaluateContext(ExprMemo);
    }

    /// <inheritdoc/>
    protected override Cost VisitLeafBaseFunction(BaseFunction expr) => Cost.Zero;

    /// <inheritdoc/>
    protected override Cost VisitLeafConst(Const expr) => Cost.Zero;

    /// <inheritdoc/>
    protected override Cost VisitLeafMarker(Marker expr) => ExprMemo[expr.Target];

    /// <inheritdoc/>
    protected override Cost VisitLeafNone(None expr) => Cost.Zero;

    /// <inheritdoc/>
    protected override Cost VisitLeafCall(Call expr)
    {
        var argumentsCost = expr.Arguments.AsValueEnumerable().Select(x => ExprMemo[x]).Sum();
        _context.CurrentCall = expr;

        var targetCost = expr.Target switch
        {
            Op op => CompilerServices.EvaluateOpCost(op, _context),
            Function func => CompilerServices.EvaluateCost(func.Body),
            Fusion fusion => CompilerServices.EvaluateCost(fusion.Body),
            _ => throw new NotImplementedException(expr.Target.ToString()),
        };
        return argumentsCost + targetCost;
    }

    /// <inheritdoc/>
    protected override Cost VisitLeafOp(Op expr) => Cost.Zero;

    /// <inheritdoc/>
    protected override Cost VisitLeafTuple(IR.Tuple expr)
    {
        return expr.Fields.AsValueEnumerable().Select(x => ExprMemo[x]).Sum();
    }

    /// <inheritdoc/>
    protected override Cost VisitLeafVar(Var expr) => Cost.Zero;
}
