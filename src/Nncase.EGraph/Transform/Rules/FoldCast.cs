using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Transform.Pattern;
using static Nncase.Transform.Pattern.F.Math;
using static Nncase.Transform.Pattern.F.Tensors;
using static Nncase.Transform.Pattern.Utility;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Transform.Rule
{
    public class FoldNopCast : EGraphRule
    {
        WildCardPattern wcin = "input";
        CallPattern wccast1, wccast2;
        FoldNopCast()
        {
            wccast1 = IsCast(wcin);
            wccast2 = IsCast(wccast1);
            Pattern = wccast2;
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            var cast1 = (Cast)result.GetExpr(wccast1).Target;
            var cast2 = (Cast)result.GetExpr(wccast2).Target;
            if (cast1.NewType == cast2.NewType)
                return result.GetExpr(wcin);
            return null;
        }
    }
}