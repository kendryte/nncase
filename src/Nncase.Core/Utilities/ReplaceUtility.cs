// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using NetFabric.Hyperlinq;
using Nncase.IR;
using Fx = System.Func<Nncase.IR.Expr, Nncase.IR.Expr>;
using ParameterInfo = Nncase.IR.ParameterInfo;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Utilities;
public class ReplaceUtility
{
    public static Expr ReplaceOp(Call call, Op op)
    {
        return call with { Target = op };
    }

    public static Expr ReplaceFirst(Call call, Expr input)
    {
        return call with { Parameters = ReplaceFirst(call.Parameters, input) };
    }

    public static Expr ReplaceFirstCondWithThrow(Call call, Expr input, Func<Expr, bool> cond) =>
        ReplaceFirstCond(call, input, cond)
            .Match(i => i, () => throw new InvalidOperationException("Can't find parameter"));

    public static Option<Expr> ReplaceFirstCond(Call call, Expr input, Func<Expr, bool> cond)
    {
        for (int i = 0; i < call.Parameters.Count; i++)
        {
            var p = call.Parameters[i];
            if (cond(p))
            {
                var newCall = call with { Parameters = ReplacePos(call.Parameters, input, i) };
                return Option.Some((Expr)newCall);
            }
        }

        return Option.None;
    }

    /// <summary>
    /// make a inputCtor that receive a new input
    /// usage:
    /// Call(FakeXXX, input, otherArg1, ...)
    /// newInput => Call(op, newInput, otherArg1, ...)
    /// it's always used for Fake to NoFake Pass with IsWildcardCall.
    /// </summary>
    /// <param name="call"></param>
    /// <param name="op"></param>
    /// <returns></returns>
    public static Fx ReplaceOpAndFirst(Call call, Op op) => input =>
    {
        return call with { Target = op, Parameters = ReplaceFirst(call.Parameters, input) };
    };

    public static T[] ReplacePos<T>(IReadOnlyList<T> arr, T v, int i)
    {
        var array = arr.ToArray();
        return array[..i].Concat(new[] { v }).Concat(array[(i + 1)..]).ToArray();
    }

    public static Expr ReplacePos(Call call, Expr input, int i, PatternMatch.MatchOptions matchOptions)
    {
        return call with
        {
            Parameters = ReplacePos(
                call.Parameters.Select(p =>
            {
                matchOptions.TryUpdateWithRewrite(ref p);
                return p;
            }).ToList(), input, i),
        };
    }

    public static T[] ReplaceFirst<T>(IReadOnlyList<T> arr, T v)
    {
        return ReplacePos(arr, v, 0);
    }

    public static T[] ReplaceMulti<T>(IReadOnlyList<T> arr, params (ParameterInfo, T)[] valueAndPosition)
    {
        var data = arr.ToArray();
        foreach (var (parameterInfo, v) in valueAndPosition)
        {
            data[parameterInfo.Index] = v;
        }

        return data;
    }

    /// <summary>
    /// Replace call params with posAndValue.
    /// It is designed to easier to see the difference between rewrite before and rewrite after
    /// e.g.
    /// before:
    /// call = Reduce(reduceOp, input, axis, initValue, keepDims)
    /// call:
    /// ReplaceParams(call,
    ///     (Nncase.IR.Math.Reduce.InitValue, newInitValue),
    ///     (Nncase.IR.Math.Reduce.Axis, newAxis)
    /// )
    /// after:
    /// call == Reduce(reduceOp, input, newAxis, initValue, keepDims)
    ///
    /// posAndValue is not required to be in order
    ///
    /// warning: call which returned should be type infer, because of with should keep the type infer.
    /// </summary>
    /// <param name="call"></param>
    /// <param name="posAndValue"></param>
    /// <returns></returns>
    public static Call ReplaceParams(Call call, params (ParameterInfo, Expr)[] posAndValue)
    {
        return call with { Parameters = ReplaceMulti(call.Parameters, posAndValue) };
    }

    /// <summary>
    /// find the old input in old args and replace it with new_input.
    /// </summary>
    /// <param name="list">matched old args.</param>
    /// <param name="target">matched old input.</param>
    /// <param name="value">created new_input.</param>
    /// <returns>new args list.</returns>
    /// <exception cref="InvalidOperationException"></exception>
    public static List<Expr> ReplaceParams(IReadOnlyList<Expr> list, Expr target, Expr value)
    {
        return ReplaceParams(list, new List<(Expr, Expr)>() { (target, value) });
    }

    /// <summary>
    ///  find the old input in old args and replace it with new_input.
    /// </summary>
    /// <param name="list">matched exprsession list.</param>
    /// <param name="pairs">target value pair.</param>
    /// <returns>new args list.</returns>
    /// <exception cref="InvalidOperationException"></exception>
    public static List<Expr> ReplaceParams(IReadOnlyList<Expr> list, IReadOnlyList<(Expr target, Expr value)> pairs)
    {
        var new_args = new List<Expr>(list);

        Dictionary<int, Expr> candidates = new();
        for (int i = 0; i < list.Count; i++)
        {
            for (int j = 0; j < pairs.Count; j++)
            {
                if (object.ReferenceEquals(new_args[i], pairs[j].target))
                {
                    if (!candidates.TryGetValue(i, out var last_matched))
                    {
                        last_matched = pairs[j].value;
                        candidates.Add(i, last_matched);
                    }

                    if (!object.ReferenceEquals(last_matched, pairs[j].value))
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
        return new_args;
    }

    public static Call ReplaceOpAndParams(Call call, Op op, params (ParameterInfo, Expr)[] posAndValue)
    {
        return call with { Target = op, Parameters = ReplaceMulti(call.Parameters, posAndValue) };
    }

    public static Expr ReplaceTarget(Expr root, Expr target, Expr expr, PatternMatch.MatchOptions matchOptions) =>
        ReplaceTargetImpl(root, target, expr, matchOptions)
            .Match(
                x => x,
                () => throw new InvalidOperationException("target not found"));

    private static Option<Expr> ReplaceTargetImpl(Expr root, Expr target, Expr expr, PatternMatch.MatchOptions matchOptions)
    {
        if (root == target)
        {
            return Option.Some(expr);
        }

        if (root is not Call)
        {
            return Option.None;
        }

        var rootCall = (Call)root;
        for (var i = 0; i < rootCall.Parameters.Count; i++)
        {
            var param = rootCall.Parameters[i];
            matchOptions.TryUpdateWithRewrite(ref param);
            var e = ReplaceTargetImpl(param, target, expr, matchOptions);
            if (e.IsSome)
            {
                return Option.Some(ReplacePos(rootCall, e.Value, i, matchOptions));
            }
        }

        return Option.None;
    }
}
