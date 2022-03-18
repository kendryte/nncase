// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Transform.Rules.Neutral;

/// <summary>
/// Fold two <see cref="IR.Tensors.Transpose"/>.
/// </summary>
[RuleGenerator]
public class FoldTwoTransposes : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsTranspose(
        IsTranspose(IsWildcard("input"), IsWildcard("perm1") with { TypePattern = HasRank() }),
        IsWildcard("perm2") with { TypePattern = HasRank() });

    private Expr? GetReplace(Expr input, Expr perm1, Expr perm2)
    {
        if (perm1.CheckedShape.Rank is int rank && rank == perm2.CheckedShape.Rank)
        {
            var newPerm = new Expr[rank];
            for (int i = 0; i < newPerm.Length; i++)
            {
                newPerm[i] = perm1[perm2[i]];
            }

            return Transpose(tp1.Input(), Const.FromTensor(perm));
        }

        return null;
    }
}

public class FoldNopTranspose : IRewriteRule
{
    TransposeWrapper tp;
    public FoldNopTranspose()
    {
        Pattern = tp = Transpose(IsWildcard(), IsConstIntTensor());
    }

    public override Expr? GetReplace(IMatchResult result)
    {
        tp.Bind(result);
        var perm = tp.Perm<TensorConst>().Value.Cast<int>();
        if (Enumerable.Range(0, (int)perm.Length).All(dim => perm[dim] == dim))
        {
            return tp.Input();
        }

        return null;
    }
}

public class TransposeToReshape : IRewriteRule
{
    TransposeWrapper tp;
    public TransposeToReshape()
    {
        Pattern = tp = Transpose(IsWildcard(), IsConstIntTensor());
    }

    public override Expr? GetReplace(IMatchResult result)
    {
        tp.Bind(result);
        var perm = tp.Perm<TensorConst>().Value.Cast<int>();
        var in_shape = tp.Input().CheckedShape;
        int last_sig_dim = 0;
        for (int i = 0; i < perm.Length; i++)
        {
            var i_dim = perm[i];
            if (in_shape[i].FixedValue != 1)
            {
                if (i_dim < last_sig_dim)
                {
                    return null;
                }

                last_sig_dim = i_dim;
            }
        }

        var outshape = result[Pattern].CheckedShape;
        return Reshape(tp.Input(), Const.FromShape(outshape));
    }
}
