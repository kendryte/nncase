// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using static Nncase.Pattern.Utility;
using Nncase.Pattern.Tensors;
using Nncase.Pattern.NN;
using Nncase.Pattern.Math;
using Nncase.Pattern;
using Nncase.IR.Tensors;
using Nncase.IR.NN;
using Nncase.IR.Math;
using Nncase;
using static Nncase.Pattern.F.Tensors;

namespace Nncase.Pattern
{
    public static partial class Utility
    {
        /// <summary>
        /// Get Alternative pattern.
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns>OrPattern.</returns>
        public static OrPattern IsAlt(ExprPattern lhs, ExprPattern rhs) => new OrPattern(lhs, rhs);

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

        public static ReduceWrapper IsReduce(Func<ReduceOp, bool> Cond, ExprPattern Input, ExprPattern Axis, ExprPattern InitValue, ExprPattern KeepDims) => IsReduce(x => Cond(x.ReduceOp), Input, Axis, InitValue, KeepDims);

        public static ReduceWrapper IsReduce(ReduceOp opType, ExprPattern Input, ExprPattern Axis, ExprPattern InitValue, ExprPattern KeepDims) => IsReduce(x => x == opType, Input, Axis, InitValue, KeepDims);

        public static ReduceWrapper IsReduce(ExprPattern Input, ExprPattern Axis, ExprPattern InitValue, ExprPattern KeepDims) => IsReduce((ReduceOp x) => true, Input, Axis, InitValue, KeepDims);

        public static PadWrapper IsPad(Func<PadMode, bool> cond, ExprPattern input, ExprPattern pads, ExprPattern value) => new PadWrapper(new CallPattern(new PadPattern(pad => cond(pad.PadMode)), input, pads, value));

        public static PadWrapper IsPad(ExprPattern input, ExprPattern pads, PadMode mode, ExprPattern value) => IsPad(x => x == mode, input, pads, value);

        public static PadWrapper IsPad(ExprPattern input, ExprPattern pads, ExprPattern value) =>
        IsPad((PadMode padmode) => true, input, pads, value);

        public static CastWrapper IsCast(Func<DataType, bool> Cond, ExprPattern input) => new CastWrapper(new CallPattern(new CastPattern((Cast x) => Cond(x.NewType)), input));

        public static CastWrapper IsCast(ExprPattern input) =>
        IsCast(x => true, input);

        public static QuantizeWrapper IsQuantize(Func<DataType, bool> Cond, ExprPattern input, ExprPattern zeroPoint, ExprPattern scale) => new QuantizeWrapper(new CallPattern(new QuantizePattern(x => Cond(x.TargetType)), input, zeroPoint, scale));

        public static QuantizeWrapper IsQuantize(ExprPattern input) => IsQuantize(x => true, input, IsConst(), IsConst());

        public static DeQuantizeWrapper IsDeQuantize(Func<DataType, bool> Cond, ExprPattern input, ExprPattern zeroPoint, ExprPattern scale) => new DeQuantizeWrapper(new CallPattern(new DeQuantizePattern(x => Cond(x.TargetType)), input, zeroPoint, scale));

        public static DeQuantizeWrapper IsDeQuantize(ExprPattern input) => IsDeQuantize(x => true, input, IsConst(), IsConst());

        public static SliceWrapper IsSlice(ExprPattern input) => Slice(input, IsConstIntTensor(), IsConstIntTensor(), IsConstIntTensor(), IsConstIntTensor());

        public static SliceWrapper IsSlice(ExprPattern input, ExprPattern begins, ExprPattern ends) => Slice(input, begins, ends, IsConstIntTensor(), IsConstIntTensor());

        public static Conv2DWrapper IsConv2D(ExprPattern input, ExprPattern weights, ExprPattern bias, PadMode padMode) => new Conv2DWrapper(new CallPattern(new Conv2DPattern(x => x.PadMode == padMode), input, weights, bias, IsConst(), IsConst(), IsConst()));

        public static Conv2DWrapper IsConv2D(ExprPattern input, ExprPattern weights, ExprPattern bias) => new Conv2DWrapper(new CallPattern(new Conv2DPattern(x => true), input, weights, bias, IsConst(), IsConst(), IsConst(), IsConst()));

        public static Conv2DWrapper IsConv2D(ExprPattern input) => IsConv2D(input, IsWildCard(), IsWildCard());

        public static Conv2DWrapper IsConv2D(ExprPattern input, PadMode padMode) => IsConv2D(input, IsWildCard(), IsWildCard(), padMode);

        public static ClampWrapper IsClamp(ExprPattern input) => new ClampWrapper(new CallPattern(new ClampPattern(), input, IsConst(), IsConst()));

        public static ResizeImageWrapper IsResize(ExprPattern input, ExprPattern newSize, ExprPattern alignCorners, ExprPattern halfPixelCenters) => new ResizeImageWrapper(new CallPattern(new ResizeImagePattern(x => true), input, newSize, alignCorners, halfPixelCenters));

        public static ResizeImageWrapper IsResize(ExprPattern input, ExprPattern newSize) => IsResize(input, newSize, IsConst(), IsConst());
    }
}