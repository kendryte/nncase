// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using System.Collections.Immutable;
using System.Collections.Generic;
using System;
using static Nncase.Pattern.Utility;
using static Nncase.Pattern.F.Tensors;
using static Nncase.Pattern.F.Math;
using static Nncase.IR.TypePatternUtility;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.Math;
using Nncase.Pattern;
using Nncase.IR.Tensors;
using Nncase.IR.Math;
using Nncase.IR;

namespace Nncase.Transform.Rule
{
    public class FoldReshape : IRewriteRule
    {
        WildCardPattern wcin = "input";
        WildCardPattern shape1 = "sp1", shape2 = "sp2";

        public FoldReshape()
        {
            Pattern = Reshape(Reshape(wcin, shape1), shape2);
        }

        public override Expr? GetReplace(IMatchResult result)
        {
            return Reshape(result[wcin], result[shape2]);
        }
    }

    public class FoldNopReshape : IRewriteRule
    {
        WildCardPattern wcin = "input";
        TensorConstPattern wcshape = IsTensorConst(IsIntegral());

        public FoldNopReshape()
        {
            Pattern = Reshape(wcin, wcshape);
        }

        public override Expr? GetReplace(IMatchResult result)
        {
            var input = result[wcin];
            var shape = result[wcshape].Value.Cast<int>();
            var type = input.CheckedType;
            if (type is TensorType ttype)
            {
                if (!ttype.Shape.IsFixed)
                {
                    return null;
                }

                // ttype.Shape
                var targetShape = new Shape(shape);
                if (ttype.Shape == targetShape)
                {
                    return input;
                }
            }

            return null;
        }
    }
}