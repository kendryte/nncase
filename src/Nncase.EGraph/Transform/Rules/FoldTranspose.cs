using System.Linq;
using Nncase.IR;
using Nncase.Transform.Pattern.Tensors;
using static Nncase.Transform.Pattern.Utility;
using System.Numerics.Tensors;
using static Nncase.IR.F.Tensors;
using static Nncase.Transform.Pattern.F.Tensors;

namespace Nncase.Transform.Rule
{

    public class FoldTranspose : EGraphRule
    {
        TransposeWrapper tp1, tp2;
        public FoldTranspose()
        {
            tp1 = Transpose(IsWildCard(), IsConstIntTensor());
            tp2 = Transpose(tp1, IsConstIntTensor());
            Pattern = tp2;
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            tp1.Bind(result);
            tp2.Bind(result);
            var perm1 = tp1.Perm<Const>().ToTensor<int>();
            var perm2 = tp2.Perm<Const>().ToTensor<int>();
            if (perm1.Rank == perm2.Rank)
            {
                var perm = new DenseTensor<int>(perm1.Dimensions);
                for (int i = 0; i < perm1.Rank; i++)
                {
                    perm[i] = perm1[perm2[i]];
                }
                return Transpose(tp1.Input(), Const.FromTensor<int>(perm));
            }
            return null;
        }
    }


    public class FoldNopTranspose : EGraphRule
    {
        TransposeWrapper tp;
        public FoldNopTranspose()
        {
            Pattern = tp = Transpose(IsWildCard(), IsConstIntTensor());
        }
        public override Expr? GetRePlace(EMatchResult result)
        {
            tp.Bind(result);
            var perm = tp.Perm<Const>().ToTensor<int>();
            if (Enumerable.Range(0, (int)tp.Input().CheckedShape.Rank).All(dim => perm[dim] == dim))
            {
                return tp.Input();
            }
            return null;
        }
    }

    public class TransposeToReshape : EGraphRule
    {
        TransposeWrapper tp;
        public TransposeToReshape()
        {
            Pattern = tp = Transpose(IsWildCard(), IsConstIntTensor());
        }
        public override Expr? GetRePlace(EMatchResult result)
        {
            tp.Bind(result);
            var perm = tp.Perm<Const>().ToTensor<int>();
            var in_shape = tp.Input().CheckedShape;
            int last_sig_dim = 0;
            for (int i = 0; i < perm.Length; i++)
            {
                var i_dim = perm[i];
                if (in_shape[i].FixedValue != 1)
                {
                    if (i_dim < last_sig_dim)
                        return null;
                    last_sig_dim = i_dim;
                }
            }
            var outshape = result[Pattern].CheckedShape;
            return Reshape(tp.Input(), Const.FromShape(outshape));
        }
    }
}