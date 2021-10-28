// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.Transform.Pattern.Math;


namespace Nncase.Transform.Pattern
{
    public sealed record CallPattern(ID Id, ExprPattern Target, VArgsPattern Parameters) : ExprPattern(Id)
    {

        public CallPattern(Call call) : this(Utility.GetID(), (ExprPattern)call.Target, new VArgsPattern(call.Parameters)) { }

        public bool MatchLeaf(Call call)
        {
            return MatchCheckedType(call);
        }

        public CallPattern(ID Id, ExprPattern target, params ExprPattern[] parameters)
            : this(Id, target, new VArgsPattern(parameters))
        {
        }
    }

    public static partial class Utility
    {

        public static CallPattern IsCall(ID Id, ExprPattern Target, VArgsPattern Parameters) => new CallPattern(Id, Target, Parameters);

        public static CallPattern IsCall(ExprPattern Target, VArgsPattern Parameters) => new CallPattern(GetID(), Target, Parameters);

        public static CallPattern IsCall(ID Id, ExprPattern Target, params ExprPattern[] Parameters) => new CallPattern(Id, Target, Parameters);

        public static CallPattern IsCall(ExprPattern Target, params ExprPattern[] Parameters) => new CallPattern(GetID(), Target, Parameters);


        public static CallPattern IsBinary(ID Id, Func<BinaryOp, bool> OpTypeCond, ExprPattern lhs, ExprPattern rhs) =>
          new CallPattern(Id, new BinaryPattern(binary => OpTypeCond(binary.BinaryOp)), lhs, rhs);

        public static CallPattern IsBinary(Func<BinaryOp, bool> OpTypeCond, ExprPattern lhs, ExprPattern rhs) => IsBinary(GetID(), OpTypeCond, lhs, rhs);
        public static CallPattern IsBinary(ID Id, BinaryOp opType, ExprPattern lhs, ExprPattern rhs) =>
          IsBinary(Id, binaryOp => opType == binaryOp, lhs, rhs);

        public static CallPattern IsBinary(BinaryOp opType, ExprPattern lhs, ExprPattern rhs) => IsBinary(GetID(), opType, lhs, rhs);

        public static CallPattern IsBinary(ID Id, ExprPattern lhs, ExprPattern rhs) => IsBinary(Id, binaryOp => true, lhs, rhs);

        public static CallPattern IsBinary(ExprPattern lhs, ExprPattern rhs) => IsBinary(GetID(), lhs, rhs);

        public static CallPattern IsUnary(ID Id, Func<UnaryOp, bool> OpTypeCond, ExprPattern input) =>
          new CallPattern(Id, new UnaryPattern(unary => OpTypeCond(unary.UnaryOp)), input);

        public static CallPattern IsUnary(Func<UnaryOp, bool> OpTypeCond, ExprPattern input) => IsUnary(GetID(), OpTypeCond, input);

        public static CallPattern IsUnary(ID Id, UnaryOp opType, ExprPattern input) => IsUnary(Id, unaryOp => opType == unaryOp, input);

        public static CallPattern IsUnary(UnaryOp opType, ExprPattern input) => IsUnary(GetID(), opType, input);

        public static CallPattern IsUnary(ID Id, ExprPattern input) => IsUnary(Id, unaryOp => true, input);

        public static CallPattern IsUnary(ExprPattern input) => IsUnary(GetID(), input);

    }
}