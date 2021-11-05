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
    public class FoldReshape : EGraphRule
    {
        WildCardPattern wcin = "input";
        WildCardPattern shape1 = "sp1", shape2 = "sp2";

        FoldReshape()
        {
            Pattern = ReShape(ReShape(wcin, shape1), shape2);
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            return ReShape(result[wcin], result[shape2]);
        }
    }

    public class FoldNopReshape : EGraphRule
    {
        WildCardPattern wcin = "input";
        ConstPattern wcshape = IsConst(IsTensor() & IsIntegral());

        FoldNopReshape()
        {
            Pattern = ReShape(wcin, wcshape);
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            var input = result[wcin];
            var shape = result[wcshape].ToTensor<int>();
            var type = input.CheckedType;
            if (type is TensorType ttype)
            {
                if (!ttype.Shape.IsFixed)
                    return null;
                // ttype.Shape
                var targetShape = new Shape(shape.ToImmutableArray());
                if (ttype.Shape == targetShape)
                    return input;
            }
            return null;
        }
    }
}