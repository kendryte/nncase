// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Transform.Pattern.Math;
using Nncase.Transform.Pattern.Tensors;


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


        public static CallPattern IsReduce(Func<Reduce, bool> Cond, ExprPattern Input, ExprPattern Axis, ExprPattern InitValue, ExprPattern KeepDims) => new CallPattern(new ReducePattern(Cond), Input, Axis, InitValue, KeepDims);

        public static CallPattern IsReduce(Func<ReduceOp, bool> Cond, ExprPattern Input, ExprPattern Axis, ExprPattern InitValue, ExprPattern KeepDims) => IsReduce(x => Cond(x.reduceOp), Input, Axis, InitValue, KeepDims);

        public static CallPattern IsReduce(ReduceOp opType, ExprPattern Input, ExprPattern Axis, ExprPattern InitValue, ExprPattern KeepDims) => IsReduce(x => x == opType, Input, Axis, InitValue, KeepDims);

        public static CallPattern IsReduce(ExprPattern Input, ExprPattern Axis, ExprPattern InitValue, ExprPattern KeepDims) => IsReduce((ReduceOp x) => true, Input, Axis, InitValue, KeepDims);

        public static CallPattern IsPad(Func<PadMode, bool> cond, ExprPattern input, ExprPattern pads, ExprPattern value) => new CallPattern(new PadPattern(pad => cond(pad.padMode)), input, pads, value);

        public static CallPattern IsPad(ExprPattern input, ExprPattern pads, ExprPattern value) =>
        IsPad((PadMode padmode) => true, input, pads, value);

        public static CallPattern IsCast(Func<DataType, bool> Cond, ExprPattern input) => new CallPattern(new CastPattern(Cond), input);

        public static CallPattern IsCast(ExprPattern input) =>
        IsCast(x => true, input);

        public static CallPattern IsQuantize(Func<DataType, bool> Cond, ExprPattern input, ExprPattern quantParam) => new CallPattern(new QuantizePattern(x => Cond(x.TargetType)), input, quantParam);

        public static CallPattern IsQuantize(ExprPattern input, ExprPattern quantParam) => IsQuantize(x => true, input, quantParam);

        public static CallPattern IsDeQuantize(Func<DataType, bool> Cond, ExprPattern input, ExprPattern quantParam) => new CallPattern(new DeQuantizePattern(x => Cond(x.TargetType)), input, quantParam);

        public static CallPattern IsDeQuantize(ExprPattern input, ExprPattern quantParam) => IsDeQuantize(x => true, input, quantParam);
    }
}