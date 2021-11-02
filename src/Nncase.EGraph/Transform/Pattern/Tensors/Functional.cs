// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Transform.Pattern.Tensors;
using static Nncase.Transform.Pattern.Utility;


namespace Nncase.Transform.Pattern.F
{
    public static class Tensor
    {
        public static CallPattern Transpose(ExprPattern input, ExprPattern perm) => new CallPattern(new TransposePattern(x => true), input, perm);

        public static CallPattern Concat(TuplePattern input, ExprPattern axis) => new CallPattern(new ConcatPattern(x => true), input, axis);

        public static CallPattern Pad(ExprPattern Input, TuplePattern Pads, ExprPattern Mode, ExprPattern Value) => new CallPattern(new PadPattern(), Input, Pads, Mode, Value);

        public static CallPattern ReduceMean(ExprPattern Input, ExprPattern Axis, ExprPattern InitValue, ExprPattern KeepDims) => new CallPattern(new ReducePattern(ReduceOp.Mean), Input, Axis, InitValue, KeepDims);

        public static CallPattern ReduceMin(ExprPattern Input, ExprPattern Axis, ExprPattern InitValue, ExprPattern KeepDims) => new CallPattern(new ReducePattern(ReduceOp.Min), Input, Axis, InitValue, KeepDims);

        public static CallPattern ReduceMax(ExprPattern Input, ExprPattern Axis, ExprPattern InitValue, ExprPattern KeepDims) => new CallPattern(new ReducePattern(ReduceOp.Max), Input, Axis, InitValue, KeepDims);

        public static CallPattern ReduceSum(ExprPattern Input, ExprPattern Axis, ExprPattern InitValue, ExprPattern KeepDims) => new CallPattern(new ReducePattern(ReduceOp.Sum), Input, Axis, InitValue, KeepDims);

        public static CallPattern Squeeze(ExprPattern input, ExprPattern dims) => new CallPattern(new SqueezePattern(), input, dims);

        public static CallPattern ReShape(ExprPattern input, ExprPattern shape) => new CallPattern(new ReshapePattern(), input, shape);
    }

}
