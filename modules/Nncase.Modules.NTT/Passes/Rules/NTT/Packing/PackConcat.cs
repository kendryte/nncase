// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

[RuleGenerator]
public sealed partial class PackConcatPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
    PatternMatch.F.Tensors.IsPack(
            "pack",
            "caller",
            _ => true,
            IsConcat(
                "concat",
                "callee",
                _ => true,
                IsTuple(null, IsVArgsRepeat("tupleInputs", () => IsWildcard(null, e => e is Expr) with { TypePattern = HasRankedShape() }))));

    private Expr? GetReplace(Pack pack, Concat concat, Call caller, Call callee, IReadOnlyList<BaseExpr> tupleInputs)
    {
        var packAxisIndex = pack.Axes.IndexOf(concat.Axis);
        if (packAxisIndex != -1
            && tupleInputs.Any(input => !Dimension.TryDivExactly(input.CheckedShape[concat.Axis], pack.Lanes[packAxisIndex], out _)))
        {
            return null; // If any input is not packable, we cannot replace the concat with a pack.
        }

        var newInputs = tupleInputs.Select(input => caller.WithArguments([(Pack.Input, input)])).ToArray();
        return callee.WithArguments([(Concat.Input, new IR.Tuple(newInputs))]);
    }
}

[RuleGenerator]
public sealed partial class ConcatUnpackPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
    IsConcat(
        "concat",
        "caller",
        _ => true,
        IsTuple(null, IsVArgsRepeat("tupleInputs", exprs =>
        {
            var patterns = new Pattern[exprs.Length];
            for (var i = 0; i < patterns.Length; i++)
            {
                patterns[i] = IsAlt(
                    PatternMatch.F.Tensors.IsUnpack($"unpack_{i}", $"callee_{i}", _ => true, IsWildcard($"input_{i}")),
                    IsWildcard($"input_{i}") with { TypePattern = !IsVector() });
            }

            return patterns;
        })));

    private Expr? GetReplace(Concat concat, Call caller, IReadOnlyList<BaseExpr> tupleInputs, IMatchResult result)
    {
        var firstUnpack = (from i in Enumerable.Range(0, tupleInputs.Count)
                           let unpack = result.GetValueOrDefault($"unpack_{i}") as Unpack
                           where unpack is not null
                           select unpack).FirstOrDefault();
        if (firstUnpack is null)
        {
            return null; // If no unpack is found, we cannot replace the concat with a pack.
        }

        var newInputs = Enumerable.Range(0, tupleInputs.Count)
            .Select(i => (Expr)result[$"input_{i}"])
            .ToArray();
        for (int i = 0; i < newInputs.Length; i++)
        {
            var unpack = result.GetValueOrDefault($"unpack_{i}") as Unpack;
            if (unpack is not null)
            {
                if (unpack.Axes != firstUnpack.Axes ||
                    unpack.Lanes != firstUnpack.Lanes)
                {
                    return null; // If any unpack has different axes or lanes, we cannot replace the concat.
                }
            }
            else
            {
                for (int j = 0; j < firstUnpack.Axes.Count; j++)
                {
                    if (!Dimension.TryDivExactly(newInputs[i].CheckedShape[firstUnpack.Axes[j]], firstUnpack.Lanes[j], out _))
                    {
                        return null; // If any input is not unpackable, we cannot replace the concat.
                    }
                }

                newInputs[i] = IR.F.Tensors.Pack(
                    newInputs[i],
                    firstUnpack.Lanes.ToArray(),
                    firstUnpack.Axes.ToArray());
            }
        }

        return IR.F.Tensors.Unpack(caller.WithArguments([(Concat.Input, new IR.Tuple(newInputs))]), firstUnpack.Lanes.ToArray(), firstUnpack.Axes.ToArray());
    }
}
