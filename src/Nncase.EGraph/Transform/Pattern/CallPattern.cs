// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.Transform.Pattern.Math;

namespace Nncase.Transform.Pattern
{
    public sealed record CallPattern(ExprPattern Target, IRArray<ExprPattern> Parameters) : ExprPattern
    {
        // public override bool Match(Call call)
        // {
        //     if (Parameters.Count != call.Parameters.Count)
        //     {
        //         return false;
        //     }
        //     if (!Target.Match(call.Target))
        //     {
        //         return false;
        //     }
        //     foreach (var (ppat, p) in Parameters.Zip(call.Parameters))
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
            return (Parameters.Count == call.Parameters.Count) && MatchCheckedType(call);
        }

        public CallPattern(ExprPattern target, params ExprPattern[] parameters)
            : this(target, ImmutableArray.Create(parameters))
        {
        }

    }

    public static partial class Functional
    {
        public static CallPattern IsBinary(Func<BinaryOp, bool> OpTypeCond, ExprPattern lhs, ExprPattern rhs) =>
          new CallPattern(new BinaryPattern(OpTypeCond), lhs, rhs);

        public static CallPattern IsBinary(BinaryOp opType, ExprPattern lhs, ExprPattern rhs) =>
          new CallPattern(new BinaryPattern(opType), lhs, rhs);

        public static CallPattern IsUnary(Func<UnaryOp, bool> OpTypeCond, ExprPattern input) =>
          new CallPattern(new UnaryPattern(OpTypeCond), input);

        public static CallPattern IsUnary(UnaryOp opType, ExprPattern input) =>
          new CallPattern(new UnaryPattern(opType), input);

    }
}