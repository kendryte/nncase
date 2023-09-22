// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Fold nop <see cref="IR.Tensors.Transpose"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldNopTranspose : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsTranspose(
        IsWildcard("input"),
        IsTensorConst("perm", IsIntegral()));

    private Expr? GetReplace(Expr input, Tensor<int> perm)
    {
        for (int i = 0; i < perm.Length; i++)
        {
            if (perm[i] != i)
            {
                return null;
            }
        }

        return input;
    }
}

/// <summary>
/// Fold two <see cref="IR.Tensors.Transpose"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldTwoTransposes : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsTranspose(
        IsTranspose(IsWildcard("input"), IsWildcard("perm1") with { TypePattern = HasRank() }),
        IsWildcard("perm2") with { TypePattern = HasRank() });

    private Expr? GetReplace(Expr input, Expr perm1, Expr perm2)
    {
        if (perm1.CheckedShape.Rank is int rank && rank == perm2.CheckedShape.Rank)
        {
            if (perm1 is TensorConst cperm1 && perm2 is TensorConst cperm2)
            {
                var p1 = cperm1.Value.ToArray<int>();
                var p2 = cperm2.Value.ToArray<int>();
                var np = new int[p2.Length];
                bool is_nop = true;
                for (int i = 0; i < p2.Length; i++)
                {
                    np[i] = p1[p2[i]];
                    is_nop &= np[i] == i;
                }

                if (is_nop)
                {
                    return input;
                }

                return IR.F.Tensors.Transpose(input, np);
            }

            var newPerm = new Expr[perm2.CheckedShape[0].FixedValue];
            for (int i = 0; i < newPerm.Length; i++)
            {
                newPerm[i] = perm1[perm2[i]];
            }

            return Transpose(input, Stack(new Tuple(newPerm), 0));
        }

        return null;
    }
}

/// <summary>
/// Replace <see cref="IR.Tensors.Transpose"/> with <see cref="IR.Tensors.Reshape"/>.
/// </summary>
[RuleGenerator]
public sealed partial class TransposeToReshape : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsTranspose(
        target_name: null,
        call_name: "tp",
        IsWildcard("input") with { TypePattern = HasRank() },
        IsTensorConst("perm", IsIntegral())) with
    { TypePattern = HasFixedShape() };

    private Expr? GetReplace(Expr input, Expr tp, Tensor<int> perm, RunPassContext context)
    {
        if (input.CheckedShape.Rank <= 1)
        {
            return null;
        }

        // If all significant dims remains ascending order, it can be converted to a reshape.
        var inShape = input.CheckedShape;
        var sigAxes = new HashSet<int>();
        for (int i = 0; i < inShape.Rank; i++)
        {
            if (inShape[i] != 1)
            {
                sigAxes.Add(i);
            }
        }

        var lastPerm = int.MinValue;
        for (int i = 0; i < perm.Length; i++)
        {
            var value = perm[i];
            if (sigAxes.Contains(value))
            {
                if (value > lastPerm)
                {
                    lastPerm = value;
                }
                else
                {
                    return null;
                }
            }
        }

        context.MatchOptions.SuppressPattern(tp, Pattern);
        return Reshape(input, tp.CheckedShape);
    }
}
