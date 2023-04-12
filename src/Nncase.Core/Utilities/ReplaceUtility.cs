// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
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
    public static Expr[] ReplaceItems(IReadOnlyList<Expr> list, params (Expr Target, Expr Value)[] pairs)
    {
        if (pairs.Length == 0)
        {
            return list.ToArray();
        }

        var new_args = new List<Expr>(list);

        Dictionary<int, Expr> candidates = new();
        for (int i = 0; i < list.Count; i++)
        {
            for (int j = 0; j < pairs.Length; j++)
            {
                if (object.ReferenceEquals(new_args[i], pairs[j].Target))
                {
                    if (!candidates.TryGetValue(i, out var last_matched))
                    {
                        last_matched = pairs[j].Value;
                        candidates.Add(i, last_matched);
                    }

                    if (!object.ReferenceEquals(last_matched, pairs[j].Value))
                    {
                        throw new InvalidDataException("The same arg can't replace with two new pararmeter!");
                    }
                }
            }
        }

        if (candidates.Count == 0)
        {
            throw new InvalidOperationException("Not find the replace param");
        }

        foreach (var (i, new_input) in candidates)
            new_args[i] = new_input;
        return new_args.ToArray();
    }

    /// <summary>
    /// replace items with param info.
    /// </summary>
    /// <param name="list">expr list.</param>
    /// <param name="pairs">pairs.</param>
    /// <returns>replaced list.</returns>
    public static Expr[] ReplaceItems(IReadOnlyList<Expr> list, params (IR.ParameterInfo Info, Expr Value)[] pairs)
    {
        return ReplaceItems(list, pairs.Select(p => (list[p.Info.Index], p.Value)).ToArray());
    }

    /// <summary>
    /// replace call parameters.
    /// </summary>
    /// <param name="target">new call target.</param>
    /// <param name="oldParams">old params.</param>
    /// <param name="pairs">replace pairs.</param>
    /// <returns>new call.</returns>
    public static Call ReplaceCallParams(Expr target, IReadOnlyList<Expr> oldParams, params (Expr, Expr)[] pairs)
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
    public static Call ReplaceCallParams(Expr target, IReadOnlyList<Expr> oldParams, params (IR.ParameterInfo, Expr)[] pairs)
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
    public static Call ReplaceCallFirstParam(Expr target, IReadOnlyList<Expr> oldParams, Expr expr) =>
        ReplaceCallParams(target, oldParams, (oldParams[0], expr));
}
