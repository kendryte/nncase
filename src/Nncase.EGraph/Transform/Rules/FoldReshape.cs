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
    public class FoldReshape : PatternRule
    {
        WildCardPattern wcin = "input";
        WildCardPattern shape1 = "sp1", shape2 = "sp2";

        public FoldReshape()
        {
            Pattern = Reshape(Reshape(wcin, shape1), shape2);
        }

        public override Expr? GetRePlace(IMatchResult result)
        {
            return Reshape(result[wcin], result[shape2]);
        }
    }

    public class FoldNopReshape : PatternRule
    {
        WildCardPattern wcin = "input";
        ConstPattern wcshape = IsConst(IsTensor() & IsIntegral());

        public FoldNopReshape()
        {
            Pattern = Reshape(wcin, wcshape);
        }

        public override Expr? GetRePlace(IMatchResult result)
        {
            var input = result[wcin];
            var shape = result[wcshape].ToTensor<int>();
            var type = input.CheckedType;
            if (type is TensorType ttype)
            {
                if (!ttype.Shape.IsFixed)
                    return null;
                // ttype.Shape
                var targetShape = new Shape(shape.ToArray());
                if (ttype.Shape == targetShape)
                    return input;
            }
            return null;
        }
    }
}