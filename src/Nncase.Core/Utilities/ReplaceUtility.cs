using Nncase.IR;
using Tuple = Nncase.IR.Tuple;
using ParameterInfo = Nncase.IR.ParameterInfo;
using NetFabric.Hyperlinq;
namespace Nncase.Utilities;

using Fx = Func<Expr, Expr>;

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
    /// it's always used for Fake to NoFake Rule with IsWildcardCall
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

    public static Expr ReplacePos(Call call, Expr input, int i)
    {
        return call with { Parameters = ReplacePos(call.Parameters, input, i) };
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
    /// warning: call which returned should be type infer, because of with should keep the type infer
    /// </summary>
    /// <param name="call"></param>
    /// <param name="posAndValue"></pxaram>
    /// <returns></returns>
    public static Call ReplaceParams(Call call, params (ParameterInfo, Expr)[] posAndValue)
    {
        return call with { Parameters = ReplaceMulti(call.Parameters, posAndValue) };
    }

    private static Option<Expr> ReplaceTargetImpl(Expr root, Expr target, Expr expr)
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
            var e = ReplaceTargetImpl(rootCall.Parameters[i], target, expr);
            if (e.IsSome)
            {
                return Option.Some(ReplacePos(rootCall, e.Value, i));
            }
        }
        return Option.None;
    }
    
    public static Expr ReplaceTarget(Expr root, Expr target, Expr expr) =>
        ReplaceTargetImpl(root, target, expr)
            .Match(
                x => x,
                () => throw new InvalidOperationException("target not found")
            );

}