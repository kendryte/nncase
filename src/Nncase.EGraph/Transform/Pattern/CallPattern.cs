// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public sealed record CallPattern(ExprPattern TargetPat, IRArray<ExprPattern> ParameterPats) : ExprPattern
    {
        // public override bool Match(Call call)
        // {
        //     if (ParameterPats.Count != call.Parameters.Count)
        //     {
        //         return false;
        //     }
        //     if (!TargetPat.Match(call.Target))
        //     {
        //         return false;
        //     }
        //     foreach (var (ppat, p) in ParameterPats.Zip(call.Parameters))
        //     {
        //         if (!ppat.Match(p))
        //         {
        //             return false;
        //         }
        //     }
        //     return true;
        // }

        public bool MatchLeaf(Call call)
        {
            return MatchCheckedType(call);
        }

        public CallPattern(ExprPattern target, params ExprPattern[] parameters)
            : this(target, ImmutableArray.Create(parameters))
        {
        }

    }
}