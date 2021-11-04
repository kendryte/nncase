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
    public class FoldInQuant : EGraphRule
    {
        WildCardPattern wcin = "input";

        public FoldInQuant()
        {
            var quant = IsQuantize(wcin, IsWildCard());
            var dequant = IsDeQuantize(quant, IsWildCard());
            Pattern = dequant;
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            var input = result.GetExpr(wcin);
            var output = result.GetRoot();
            bool check = (input.CheckedType, output.CheckedType) switch
            {
                (TensorType intype, TensorType outtype) => intype.DType == outtype.DType,
                (_, _) => false
            };
            if(check)
              return input;
            return null;
        }
    }

}