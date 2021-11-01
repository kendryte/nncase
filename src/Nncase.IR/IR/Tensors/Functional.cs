// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Tensors;

namespace Nncase.IR.F
{
    /// <summary>
    /// NN functional helper.
    /// </summary>
    public static class Tensors
    {
        public static Call Transpose(Expr input, Expr perm) => new Call(new Transpose(), input, perm);

        public static Call Cast(Expr input, DataType newType) => new Call(new Cast(newType), input);
        
        public static Call Concat(Tuple input, Expr axis) => new Call(new Concat(), input, axis);

        public static Call Gather(Expr input, Expr axis, Expr index) => new Call(new Gather(), input, axis, index);
        
        public static Call Pad(Expr Input, Tuple Pads, Expr Mode, Expr Value) => new Call(new Pad(), Input, Pads, Mode, Value);

        public static Call Reduce(ReduceOp reduceOp, Expr Input, Expr Axis, Expr InitValue, Expr KeepDims) => new Call(new Reduce(reduceOp), Input, Axis, InitValue, KeepDims);

        public static Call ReduceMean(Expr Input, Expr Axis, Expr InitValue, Expr KeepDims) => new Call(new Reduce(ReduceOp.Mean), Input, Axis, InitValue, KeepDims);

        public static Call ReduceMin(Expr Input, Expr Axis, Expr InitValue, Expr KeepDims) => new Call(new Reduce(ReduceOp.Min), Input, Axis, InitValue, KeepDims);

        public static Call ReduceMax(Expr Input, Expr Axis, Expr InitValue, Expr KeepDims) => new Call(new Reduce(ReduceOp.Max), Input, Axis, InitValue, KeepDims);

        public static Call ReduceSum(Expr Input, Expr Axis, Expr InitValue, Expr KeepDims) => new Call(new Reduce(ReduceOp.Sum), Input, Axis, InitValue, KeepDims);
        
        public static Call Slice(Expr input, Expr begins, Expr ends) => new Call(new Slice(), input, begins, ends);
    }
}
