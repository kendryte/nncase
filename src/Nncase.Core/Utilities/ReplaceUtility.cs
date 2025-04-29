// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Linq;
using System.Reactive;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Fx = System.Func<Nncase.IR.Expr, Nncase.IR.Expr>;
using ParameterInfo = Nncase.IR.ParameterInfo;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Utilities;

/// <summary>
/// Pattern Match Replace Utility.
/// </summary>
public static class ReplaceUtility
{
    /// <summary>
    ///  find the old input in old args and replace it with new_input.
    /// </summary>
    /// <param name="list">matched exprsession list.</param>
    /// <param name="pairs">target value pair.</param>
    /// <returns>new args list.</returns>
    /// <exception cref="InvalidOperationException">when the same target match two value.</exception>
    public static BaseExpr[] ReplaceItems(IReadOnlyList<BaseExpr> list, params (BaseExpr Target, BaseExpr Value)[] pairs)
    {
        var l = list.ToList();
        return ReplaceItems(list, pairs.Select(p => (l.IndexOf(p.Target), p.Value)).ToArray());
    }

    /// <summary>
    /// replace items with param info.
    /// </summary>
    /// <param name="list">expr list.</param>
    /// <param name="pairs">pairs.</param>
    /// <returns>replaced list.</returns>
    public static BaseExpr[] ReplaceItems(IReadOnlyList<BaseExpr> list, params (int Index, BaseExpr Value)[] pairs)
    {
        if (pairs.Length == 0)
        {
            return list.ToArray();
        }

        var new_args = list.ToArray();
        var hashset = new HashSet<int>();
        foreach (var (index, value) in pairs)
        {
            if (hashset.Add(index))
            {
                new_args[index] = value;
            }
            else
            {
                throw new InvalidDataException($"The same arg {index} can't replace with two new pararmeter!");
            }
        }

        return new_args.ToArray();
    }

    /// <summary>
    /// replace items with param info.
    /// </summary>
    /// <param name="list">expr list.</param>
    /// <param name="pairs">pairs.</param>
    /// <returns>replaced list.</returns>
    public static BaseExpr[] ReplaceItems(IReadOnlyList<BaseExpr> list, params (IR.ParameterInfo Info, BaseExpr Value)[] pairs)
    {
        return ReplaceItems(list, pairs.Select(p => (p.Info.Index, p.Value)).ToArray());
    }

    /// <summary>
    /// replace call parameters.
    /// </summary>
    /// <param name="target">new call target.</param>
    /// <param name="oldParams">old params.</param>
    /// <param name="pairs">replace pairs.</param>
    /// <returns>new call.</returns>
    public static Call ReplaceCallParams(Expr target, IReadOnlyList<BaseExpr> oldParams, params (BaseExpr, BaseExpr)[] pairs)
    {
        return new Call(target, ReplaceItems(oldParams, pairs));
    }

    /// <summary>
    /// replace the call params with parameter info.
    /// </summary>
    /// <param name="target">call target.</param>
    /// <param name="oldParams">target params.</param>
    /// <param name="pairs">the param info pair.</param>
    /// <returns>new call.</returns>
    public static Call ReplaceCallParams(Expr target, IReadOnlyList<BaseExpr> oldParams, params (IR.ParameterInfo, BaseExpr)[] pairs)
    {
        return new Call(target, ReplaceItems(oldParams, pairs));
    }

    /// <summary>
    /// replace the call params with parameter info.
    /// </summary>
    /// <param name="target">call target.</param>
    /// <param name="oldParams">target params.</param>
    /// <param name="pairs">the param info pair.</param>
    /// <returns>new call.</returns>
    public static Call ReplaceCallParams(Expr target, IReadOnlyList<BaseExpr> oldParams, params (int, BaseExpr)[] pairs)
    {
        return new Call(target, ReplaceItems(oldParams, pairs));
    }

    /// <summary>
    /// replace the first params of call with expr.
    /// </summary>
    /// <param name="target">target.</param>
    /// <param name="oldParams">oldParams.</param>
    /// <param name="expr">expr.</param>
    /// <returns>new Call.</returns>
    public static Call ReplaceCallFirstParam(Expr target, IReadOnlyList<BaseExpr> oldParams, BaseExpr expr) =>
        ReplaceCallParams(target, oldParams, (0, expr));

    /// <summary>
    /// Replace target in body with expr.
    /// </summary>
    /// <param name="body">Body.</param>
    /// <param name="target">Target.</param>
    /// <param name="expr">Expr.</param>
    /// <returns>New Body.</returns>
    public static BaseExpr ReplaceExpr(BaseExpr body, BaseExpr target, BaseExpr expr)
    {
        var mutator = new Passes.Mutators.Substitutor(e =>
        {
            if (ReferenceEquals(e, target))
            {
                return expr;
            }

            return null;
        });
        return mutator.Visit(body, Unit.Default);
    }

    public static void ReplaceAllUsesWith(BaseExpr expr, BaseExpr newOperand)
    {
        expr.ReplaceAllUsesWith(newOperand);
    }
}
