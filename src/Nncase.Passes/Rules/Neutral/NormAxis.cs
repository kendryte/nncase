// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
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
    public override CallPattern Pattern { get; } = IsGather("gather", g => g.Axis < 0, IsWildcard("input") with { TypePattern = HasRank() }, IsWildcard("index") with { TypePattern = HasRank() });

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
            ps[i] = IsWildcard(i.ToString()) with { TypePattern = HasRank() };
        }

        return ps;
    })));

    private Expr? GetReplace(IR.Tensors.Concat concat, IReadOnlyList<Expr> inputs)
    {
        return IR.F.Tensors.Concat(new IR.Tuple(inputs.ToArray()), concat.Axis + inputs[0].CheckedShape.Rank);
    }
}

[RuleGenerator]
public sealed partial class NormAxisReduce : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsReduce("reduce", "call", _ => true, IsWildcard("input") with { TypePattern = HasRank() }, IsTensorConst("axes"), IsWildcard("initValue"), IsWildcard("keepDims"));

    private Expr? GetReplace(IR.Math.Reduce reduce, Call call, Expr input, int[] axes, Expr initValue, Expr keepDims)
    {
        if (axes.Any(axis => axis < 0))
        {
            return IR.F.Tensors.Reduce(reduce.ReduceOp, input, axes.Select(axis => axis < 0 ? axis + input.CheckedShape.Rank : axis).ToArray(), initValue, keepDims);
        }

        return call;
    }
}

[RuleGenerator]
public sealed partial class NormAxisReduceArg : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsReduceArg("reduce", "call", _ => true, IsWildcard("input") with { TypePattern = HasRank() }, IsTensorConst("axis"), IsWildcard("keepDims"), IsWildcard("selectLastIndex"));

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
    public override CallPattern Pattern { get; } = IsReshape("reshape", "call", IsWildcard("input") with { TypePattern = HasFixedShape() }, IsTensorConst("newshape")) with { TypePattern = HasFixedShape() };

    private Expr? GetReplace(Call call, Expr input, int[] newshape)
    {
        if (newshape.Any(dim => dim < 0))
        {
            return IR.F.Tensors.Reshape(input, call.CheckedShape.ToValueArray());
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class NormAxisSlice : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsSlice("slice", "call", IsWildcard("input") with { TypePattern = HasFixedShape() }, IsTensorConst("begins"), IsTensorConst("ends"), IsTensorConst("axes"), IsTensorConst("strides")) with { TypePattern = HasFixedShape() };

    private Expr? GetReplace(Call call, Expr input, Expr begins, Expr ends, int[] axes, Expr strides)
    {
        if (axes.Any(dim => dim < 0))
        {
            return IR.F.Tensors.Slice(input, begins, ends, axes.Select(dim => dim < 0 ? dim + input.CheckedShape.Rank : dim).ToArray(), strides);
        }

        return null;
    }
}
