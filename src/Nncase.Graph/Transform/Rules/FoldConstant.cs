using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Pattern;
using static Nncase.Pattern.F.Math;
using static Nncase.Pattern.F.NN;
using static Nncase.Pattern.F.Tensors;
using static Nncase.Pattern.Utility;

namespace Nncase.Transform.Rule
{
    public class FoldConstCall : PatternRule
    {
        public FoldConstCall()
        {
            Pattern = IsCall(IsWildCard(), IsVArgsRepeat(() => IsAlt(IsConst(), IsConstTuple())));
        }

        public override Expr? GetRePlace(IMatchResult result)
        {
            var expr = result[Pattern];
            var tensor = expr.Eval();
            return tensor.ToConst((TensorType)expr.CheckedType!);
        }
    }

    public class FoldConstFunction : PatternRule
    {
        public FoldConstFunction()
        {
            Pattern = IsFunction(IsWildCard(), IsVArgsRepeat(() => IsAlt(IsConst(), IsConstTuple())));
        }
        public override Expr? GetRePlace(IMatchResult result) => result[Pattern].Eval().ToConst();
    }

    public class FoldShapeOp : PatternRule
    {
        WildCardPattern wc = "input";

        public FoldShapeOp()
        {
            Pattern = ShapeOp(wc);
        }

        public override Expr? GetRePlace(IMatchResult result)
        {
            return Const.FromShape(result[wc].CheckedShape);
        }
    }
}
