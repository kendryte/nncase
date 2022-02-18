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
    public class FoldNopClamp : IRewriteRule
    {
        WildcardPattern wcin = "input";
        TensorConstPattern wcmin = IsTensorConst(IsScalar());
        TensorConstPattern wcmax = IsTensorConst(IsScalar());

        public FoldNopClamp()
        {
            Pattern = Clamp(wcin, wcmin, wcmax);
        }

        public override Expr? GetReplace(IMatchResult result)
        {
            var input = result[wcin];
            var (min, max) = result[wcmin, wcmax];
            if (min.Value.ToScalar<float>() == float.MinValue &&
                max.Value.ToScalar<float>() == float.MaxValue)
            {
                return input;
            }

            return null;
        }
    }
}