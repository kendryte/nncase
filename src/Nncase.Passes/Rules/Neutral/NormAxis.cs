// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class NormAxisGather : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsGather("gather", g => g.Axis < 0, IsWildcard("input") with { TypePattern = HasRankedShape() }, IsWildcard("index") with { TypePattern = HasRankedShape() });

    private Expr? GetReplace(IR.Tensors.Gather gather, Expr input, Expr index)
    {
        return IR.F.Tensors.Gather(input, gather.Axis + input.CheckedShape.Rank, index);
    }
}

[RuleGenerator]
public sealed partial class NormAxisConcat : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsConcat("concat", op => op.Axis < 0, IsTuple(IsVArgsRepeat("inputs", inputs =>
    {
        var ps = new Pattern[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            ps[i] = IsWildcard(i.ToString()) with { TypePattern = HasRankedShape() };
        }

        return ps;
    })));

    private Expr? GetReplace(IR.Tensors.Concat concat, IReadOnlyList<BaseExpr> inputs)
    {
        return IR.F.Tensors.Concat(new IR.Tuple(inputs.ToArray()), concat.Axis + inputs[0].CheckedShape.Rank);
    }
}

[RuleGenerator]
public sealed partial class NormAxisReduce : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsReduce("reduce", "call", _ => true, IsWildcard("input") with { TypePattern = HasRankedShape() }, IsRankedShape("axes"), IsWildcard("initValue"), IsWildcard("keepDims"));

    private Expr? GetReplace(IR.Math.Reduce reduce, Call call, Expr input, Shape axes, Expr initValue, Expr keepDims)
    {
        var newAxes = axes.Select(axis => axis.IsFixed && axis.FixedValue < 0 ? axis + input.CheckedShape.Rank : axis).ToArray();
        if (newAxes.SequenceEqual(axes))
        {
            return null;
        }

        return IR.F.Tensors.Reduce(reduce.ReduceOp, input, newAxes, initValue, keepDims);
    }
}

[RuleGenerator]
public sealed partial class NormAxisReduceArg : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsReduceArg("reduce", "call", _ => true, IsWildcard("input") with { TypePattern = HasRankedShape() }, IsFixedDimension("axis"), IsWildcard("keepDims"), IsWildcard("selectLastIndex"));

    private Expr? GetReplace(IR.Math.ReduceArg reduce, Call call, Expr input, int axis, Expr keepDims, Expr selectLastIndex)
    {
        if (axis < 0)
        {
            return IR.F.Tensors.ReduceArg(reduce.ReduceArgOp, reduce.DestType, input, axis + input.CheckedShape.Rank, keepDims, selectLastIndex);
        }

        return call;
    }
}

[RuleGenerator]
public sealed partial class NormAxisReshape : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsReshape("reshape", "call", IsWildcard("input"), IsRankedShape("newShape"));

    private Expr? GetReplace(Call call, Expr input, Shape newShape)
    {
        if (newShape.Any(dim => dim.IsFixed && dim.FixedValue < 0))
        {
            return IR.F.Tensors.Reshape(input, call.CheckedShape);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class NormAxisSlice : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsSlice("slice", "call", IsWildcard("input"), IsFixedShape("begins"), IsFixedShape("ends"), IsFixedShape("axes"), IsFixedShape("strides"));

    private Expr? GetReplace(Call call, Expr input, Shape begins, long[] ends, long[] axes, Shape strides)
    {
        if (axes.Any(dim => dim < 0) || ends.Any(i => i > int.MaxValue))
        {
            axes = axes.Select(dim => dim < 0 ? dim + input.CheckedShape.Rank : dim).ToArray();
            for (int i = 0; i < axes.Length; i++)
            {
                ends[i] = ends[i] > int.MaxValue ? input.CheckedShape[(int)axes[i]].FixedValue : ends[i];
            }

            return IR.F.Tensors.Slice(input, begins, ends, axes, strides);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class NormAxisLayernorm : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
        IsLayerNorm(
        "ln",
        "call",
        _ => true,
        IsWildcard("input") with { TypePattern = HasRankedShape() },
        IsWildcard("scale") with { TypePattern = HasRankedShape() },
        IsWildcard("bias") with { TypePattern = HasRankedShape() });

    private Expr? GetReplace(Expr input, Call call, LayerNorm ln, TensorConst scale, TensorConst bias)
    {
        if (ln.Axis < 0)
        {
            return IR.F.NN.LayerNorm(ln.Axis + input.CheckedShape.Rank, ln.Epsilon, input, scale, bias, ln.UseMean, ln.ChannelFirst);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class NormAxisSoftmax : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
        IsSoftmax(
            "softmax",
            "softmaxCall",
            _ => true,
            IsWildcard("input"),
            IsFixedDimension("axis"));

    private Expr? GetReplace(Expr input, Call softmaxCall, int axis)
    {
        if (axis < 0)
        {
            return IR.F.NN.Softmax(input, axis + input.CheckedShape.Rank);
        }

        return null;
    }
}
