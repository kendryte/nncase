using System.Linq;
using System.Collections.Immutable;
using System.Collections.Generic;
using System;
using static Nncase.Pattern.Utility;
using static Nncase.Pattern.F.Tensors;
using static Nncase.Pattern.F.Math;
using static Nncase.IR.Utility;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.Math;
using Nncase.Pattern;
using Nncase.IR.Tensors;
using Nncase.IR.Math;
using Nncase.IR;

namespace Nncase.Transform.Rule
{
    public class FoldNopClamp : PatternRule
    {
        WildCardPattern wcin = "input";
        ConstPattern wcmin = IsConst(IsScalar());
        ConstPattern wcmax = IsConst(IsScalar());

        FoldNopClamp()
        {
            Pattern = Clamp(wcin, wcmin, wcmax);
        }

        public override Expr? GetRePlace(IMatchResult result)
        {
            var input = result[wcin];
            var (min, max) = result[wcmin, wcmax];
            if (min.ToScalar<float>() == Single.MinValue &&
                max.ToScalar<float>() == Single.MaxValue)
            {
                return input;
            }
            return null;
        }
    }
}