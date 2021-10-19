// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public sealed record FunctionPattern(Func<string, bool> NameCond, IRArray<ExprPattern> ParameterPats, ExprPattern BodyPat) : ExprPattern
    {
        // public override bool Match(Function func)
        // {
        //     if (!(ParameterPats.Count == func.Parameters.Count))
        //     {
        //         return false;
        //     }

        //     foreach (var (ppat, p) in ParameterPats.Zip(func.Parameters))
        //     {
        //         if (!ppat.Match(p))
        //         {
        //             return false;
        //         }
        //     }
        //     return BodyPat.Match(func.Body);
        // }

        public FunctionPattern(IRArray<ExprPattern> parameters, ExprPattern body) : this(x => true, parameters, body)
        {
        }

        public bool MatchLeaf(Function func) => NameCond(func.Name) && MatchCheckedType(func);

    }
}