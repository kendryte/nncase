// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using System.Collections.Immutable;
using System.Collections.Generic;
using System;
using static Nncase.Pattern.Utility;
using static Nncase.Pattern.F.Tensors;
using static Nncase.Pattern.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.Math;
using Nncase.Pattern;
using Nncase.IR.Tensors;
using Nncase.IR.Math;
using Nncase.IR;

namespace Nncase.Transform.Rule
{
    public class FoldNopCast : IRewriteRule
    {
        WildCardPattern wcin = "input";
        CallPattern wccast1, wccast2;
        FoldNopCast()
        {
            wccast1 = IsCast(wcin);
            wccast2 = IsCast(wccast1);
            Pattern = wccast2;
        }

        public override Expr? GetReplace(IMatchResult result)
        {
            var cast1 = (Cast)result[wccast1].Target;
            var cast2 = (Cast)result[wccast2].Target;
            if (cast1.NewType == cast2.NewType)
            {
                return result[wcin];
            }

            return null;
        }
    }
}