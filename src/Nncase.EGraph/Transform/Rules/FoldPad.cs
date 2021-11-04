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
using System.Numerics.Tensors;


namespace Nncase.Transform.Rule
{
    public class FoldNopPad : EGraphRule
    {
        WildCardPattern wcin = "input";
        ConstPattern wcpad = IsConst(IsTensor() & IsIntegral());
        public FoldNopPad()
        {
            Pattern = IsPad(wcin, wcpad, IsWildCard());
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            var pad = result.GetExpr(wcpad).ToTensor<int>();
            if (pad.All(x => x == 0))
                return result.GetExpr(wcin);
            return null;
        }
    }

    public class FoldPadPad : EGraphRule
    {
        WildCardPattern wcin = "input";
        ConstPattern wcpad1, wcpad2;
        ConstPattern wcvalue1, wcvalue2;
        CallPattern pad1, pad2;
        public FoldPadPad()
        {
            wcpad1 = IsConst(IsTensor() & IsIntegral());
            wcpad2 = wcpad1 with { };
            wcvalue1 = IsConst(IsScalar());
            wcvalue2 = wcvalue1 with { };
            pad1 = IsPad(wcin, wcpad1, wcvalue1);
            pad2 = IsPad(pad1, wcpad2, wcvalue2);
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            var (value1, value2) = result.GetExpr(wcvalue1, wcvalue2);
            var mode1 = ((Pad)result.GetExpr(pad1).Target).padMode;
            var mode2 = ((Pad)result.GetExpr(pad2).Target).padMode;
            if ((mode1 == mode2) && (mode1 != PadMode.Constant || value1.Data == value2.Data))
            {
                var (t1, t2) = (value1.ToTensor<int>(), value2.ToTensor<int>());
                var newt = new DenseTensor<int>(t1.Dimensions);
                for (int i = 0; i < t1.Dimensions[0]; i++)
                {
                    newt[i, 0] = t1[i, 0] + t2[i, 0];
                    newt[i, 1] = t1[i, 1] + t2[i, 1];
                }
                var newpad = Const.FromTensor<int>(newt);
                return Pad(result.GetExpr(wcin), newpad, mode1, value1);
            }
            return null;
        }
    }
}