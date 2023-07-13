// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;

namespace Nncase.Evaluator;

internal sealed class MetricEvaluateVisitor : ExprVisitor<Metric, Unit>
{
    private readonly MetricEvaluateContext _context;

    public MetricEvaluateVisitor()
    {
        _context = new MetricEvaluateContext(ExprMemo);
    }

    /// <inheritdoc/>
    protected override Metric VisitLeafBaseFunction(BaseFunction expr) => Metric.Zero;

    /// <inheritdoc/>
    protected override Metric VisitLeafConst(Const expr) => Metric.Zero;

    /// <inheritdoc/>
    protected override Metric VisitLeafMarker(Marker expr) => ExprMemo[expr.Target];

    /// <inheritdoc/>
    protected override Metric VisitLeafNone(None expr) => Metric.Zero;

    /// <inheritdoc/>
    protected override Metric VisitLeafCall(Call expr)
    {
        var argumentsMetric = expr.Arguments.AsValueEnumerable().Select(x => ExprMemo[x]).Sum();
        _context.CurrentCall = expr;

        var targetMetric = expr.Target switch
        {
            Op op => CompilerServices.EvaluateOpMetric(op, _context),
            _ => throw new NotImplementedException(expr.Target.ToString()),
        };
        return argumentsMetric + targetMetric;
    }

    /// <inheritdoc/>
    protected override Metric VisitLeafOp(Op expr) => Metric.Zero;

    /// <inheritdoc/>
    protected override Metric VisitLeafTuple(IR.Tuple expr)
    {
        return expr.Fields.AsValueEnumerable().Select(x => ExprMemo[x]).Sum();
    }

    /// <inheritdoc/>
    protected override Metric VisitLeafVar(Var expr) => Metric.Zero;
}
