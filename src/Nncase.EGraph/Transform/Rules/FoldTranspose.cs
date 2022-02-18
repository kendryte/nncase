// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Numerics.Tensors;
using System.Linq;
using static Nncase.Pattern.Utility;
using static Nncase.Pattern.F.Tensors;
using static Nncase.IR.F.Tensors;
using Nncase.Pattern.Tensors;
using Nncase.Pattern;
using Nncase.IR;

namespace Nncase.Transform.Rule
{
    public class FoldTranspose : IRewriteRule
    {
        TransposeWrapper tp1, tp2;
        public FoldTranspose()
        {
            tp1 = Transpose(IsWildcard(), IsConstIntTensor());
            tp2 = Transpose(tp1, IsConstIntTensor());
            Pattern = tp2;
        }

        public override Expr? GetReplace(IMatchResult result)
        {
            tp1.Bind(result);
            tp2.Bind(result);
            var perm1 = tp1.Perm<TensorConst>().Value.Cast<int>();
            var perm2 = tp2.Perm<TensorConst>().Value.Cast<int>();
            if (perm1.Rank == perm2.Rank)
            {
                var perm = new Tensor<int>(perm1.Dimensions);
                for (int i = 0; i < perm1.Length; i++)
                {
                    perm[i] = perm1[perm2[i]];
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
}