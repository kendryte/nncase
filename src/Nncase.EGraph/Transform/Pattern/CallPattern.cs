// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Transform.Pattern.NN;
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

        public ExprPattern this[ParameterInfo parameter]
        {
            get => Parameters[parameter.Index];
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


        public static BinaryWrapper IsBinary(Func<BinaryOp, bool> OpTypeCond, ExprPattern lhs, ExprPattern rhs) =>
          new BinaryWrapper(new CallPattern(new BinaryPattern(binary => OpTypeCond(binary.BinaryOp)), lhs, rhs));

        public static BinaryWrapper IsBinary(BinaryOp opType, ExprPattern lhs, ExprPattern rhs) =>
          IsBinary(binaryOp => opType == binaryOp, lhs, rhs);

        public static BinaryWrapper IsBinary(ExprPattern lhs, ExprPattern rhs) => IsBinary(binaryOp => true, lhs, rhs);

        public static UnaryWrapper IsUnary(Func<UnaryOp, bool> OpTypeCond, ExprPattern input) =>
          new UnaryWrapper(new CallPattern(new UnaryPattern(unary => OpTypeCond(unary.UnaryOp)), input));

        public static UnaryWrapper IsUnary(UnaryOp opType, ExprPattern input) => IsUnary(unaryOp => opType == unaryOp, input);
        public static UnaryWrapper IsUnary(ExprPattern input) => IsUnary(unaryOp => true, input);


        public static ReduceWrapper IsReduce(Func<Reduce, bool> Cond, ExprPattern Input, ExprPattern Axis, ExprPattern InitValue, ExprPattern KeepDims) => new ReduceWrapper(new CallPattern(new ReducePattern(Cond), Input, Axis, InitValue, KeepDims));

        public static ReduceWrapper IsReduce(Func<ReduceOp, bool> Cond, ExprPattern Input, ExprPattern Axis, ExprPattern InitValue, ExprPattern KeepDims) => IsReduce(x => Cond(x.reduceOp), Input, Axis, InitValue, KeepDims);

        public static ReduceWrapper IsReduce(ReduceOp opType, ExprPattern Input, ExprPattern Axis, ExprPattern InitValue, ExprPattern KeepDims) => IsReduce(x => x == opType, Input, Axis, InitValue, KeepDims);

        public static ReduceWrapper IsReduce(ExprPattern Input, ExprPattern Axis, ExprPattern InitValue, ExprPattern KeepDims) => IsReduce((ReduceOp x) => true, Input, Axis, InitValue, KeepDims);

        public static PadWrapper IsPad(Func<PadMode, bool> cond, ExprPattern input, ExprPattern pads, ExprPattern value) => new PadWrapper(new CallPattern(new PadPattern(pad => cond(pad.padMode)), input, pads, value));

        public static PadWrapper IsPad(ExprPattern input, ExprPattern pads, PadMode mode, ExprPattern value) => IsPad(x => x == mode, input, pads, value);

        public static PadWrapper IsPad(ExprPattern input, ExprPattern pads, ExprPattern value) =>
        IsPad((PadMode padmode) => true, input, pads, value);

        public static CastWrapper IsCast(Func<DataType, bool> Cond, ExprPattern input) => new CastWrapper(new CallPattern(new CastPattern((Cast x) => Cond(x.NewType)), input));

        public static CastWrapper IsCast(ExprPattern input) =>
        IsCast(x => true, input);

        public static QuantizeWrapper IsQuantize(Func<DataType, bool> Cond, ExprPattern input, ExprPattern quantParam) => new QuantizeWrapper(new CallPattern(new QuantizePattern(x => Cond(x.TargetType)), input, quantParam));

        public static QuantizeWrapper IsQuantize(ExprPattern input, ExprPattern quantParam) => IsQuantize(x => true, input, quantParam);

        public static DeQuantizeWrapper IsDeQuantize(Func<DataType, bool> Cond, ExprPattern input, ExprPattern quantParam) => new DeQuantizeWrapper(new CallPattern(new DeQuantizePattern(x => Cond(x.TargetType)), input, quantParam));

        public static DeQuantizeWrapper IsDeQuantize(ExprPattern input, ExprPattern quantParam) => IsDeQuantize(x => true, input, quantParam);

        public static SliceWrapper IsSlice(ExprPattern input) => F.Tensors.Slice(input, IsConstIntTensor(), IsConstIntTensor(), IsConstIntTensor(), IsConstIntTensor());

        public static SliceWrapper IsSlice(ExprPattern input, ExprPattern begins, ExprPattern ends) => F.Tensors.Slice(input, begins, ends, IsConstIntTensor(), IsConstIntTensor());

    }
}