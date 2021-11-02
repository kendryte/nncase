using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Transform.Pattern;
using static Nncase.Transform.Pattern.F.Math;
using static Nncase.Transform.Pattern.F.Tensor;
using static Nncase.Transform.Pattern.Utility;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Transform.Rule
{
    public class FoldNopClamp : EGraphRule
    {
        WildCardPattern wcin = "input";
        ConstPattern wcmin = IsConst(IsScalar());
        ConstPattern wcmax = IsConst(IsScalar());

        FoldNopClamp()
        {
            Pattern = Clamp(wcin, wcmin, wcmax);
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            var input = result.GetExpr(wcin);
            var (min, max) = result.GetExpr(wcmin, wcmax);
            if (min.ToScalar<float>() == Single.MinValue &&
                max.ToScalar<float>() == Single.MaxValue)
            {
                return input;
            }
            return null;
        }
    }
}