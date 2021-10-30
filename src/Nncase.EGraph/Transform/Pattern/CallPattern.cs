// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.Transform.Pattern.Math;


namespace Nncase.Transform.Pattern
{
    public sealed record CallPattern(ExprPattern Target, VArgsPattern Parameters) : ExprPattern
    {

        public CallPattern(Call call) : this((ExprPattern)call.Target, new FixedVArgsPattern(call.Parameters)) { }

        public bool MatchLeaf(Call call)
        {
            return MatchCheckedType(call);
        }

        public CallPattern(ExprPattern target, params ExprPattern[] parameters)
            : this(target, new FixedVArgsPattern(parameters))
        {
        }
    }

    public static partial class Utility
    {

        public static CallPattern IsCall(ExprPattern Target, VArgsPattern Parameters) => new CallPattern(Target, Parameters);

        public static CallPattern IsCall(ExprPattern Target, params ExprPattern[] Parameters) => new CallPattern(Target, Parameters);


        public static CallPattern IsBinary(Func<BinaryOp, bool> OpTypeCond, ExprPattern lhs, ExprPattern rhs) =>
          new CallPattern(new BinaryPattern(binary => OpTypeCond(binary.BinaryOp)), lhs, rhs);


        public static CallPattern IsBinary(BinaryOp opType, ExprPattern lhs, ExprPattern rhs) =>
          IsBinary(binaryOp => opType == binaryOp, lhs, rhs);

        public static CallPattern IsBinary(ExprPattern lhs, ExprPattern rhs) => IsBinary(binaryOp => true, lhs, rhs);

        public static CallPattern IsUnary(Func<UnaryOp, bool> OpTypeCond, ExprPattern input) =>
          new CallPattern(new UnaryPattern(unary => OpTypeCond(unary.UnaryOp)), input);

        public static CallPattern IsUnary(UnaryOp opType, ExprPattern input) => IsUnary(unaryOp => opType == unaryOp, input);
        public static CallPattern IsUnary(ExprPattern input) => IsUnary(unaryOp => true, input);

    }
}