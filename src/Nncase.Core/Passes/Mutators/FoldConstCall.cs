// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.IR;

namespace Nncase.Passes.Mutators;

/// <summary>
/// fold const call.
/// </summary>
public sealed class FoldConstCall : ExprRewriter
{
    private readonly Dictionary<Expr, Expr> _cseMemo = new();

    protected override Expr RewriteLeafConst(Const @const)
    {
        return CSEConst(@const);
    }

    /// <inheritdoc/>
    protected override BaseExpr RewriteLeafTuple(IR.Tuple expr)
    {
        if (IsAllConst(expr.Fields))
        {
            return CSEConst(new TupleConst(new TupleValue(expr.Fields.AsValueEnumerable().Select(x => x switch
            {
                Const c => Value.FromConst(c),
                DimConst dc => Value.FromConst(dc.Value),
                RankedShape rs => Value.FromShape(rs.ToValueArray()),
                _ => throw new NotSupportedException($"Unsupported type {x.GetType().Name} in tuple."),
            }).ToArray())));
        }

        return expr;
    }

    /// <inheritdoc/>
    protected override BaseExpr RewriteLeafCall(Call expr)
    {
        if ((expr.Target is Op op && op.CanFoldConstCall) || expr.Target is Function)
        {
            if (IsAllConst(expr.Arguments))
            {
                return CSEConst(Const.FromValue(CompilerServices.Evaluate(expr)));
            }

            if (expr.Target is IR.Tensors.GetItem && expr[IR.Tensors.GetItem.Input] is IR.Tuple tuple &&
                expr[IR.Tensors.GetItem.Index] is DimConst { Value: long index })
            {
                return tuple.Fields[(int)index];
            }
        }

        return expr;
    }

    private bool IsAllConst(ReadOnlySpan<BaseExpr> parameters) =>
      parameters.AsValueEnumerable()
        .All(e => e is Const or DimConst or RankedShape { IsFixed: true });

    private Expr CSEConst(Const c)
    {
        if (!_cseMemo.TryGetValue(c, out var result))
        {
            result = c;
            _cseMemo.Add(c, result);
        }

        return result;
    }
}
