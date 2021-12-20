// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR.F
{
    /// <summary>
    /// Tir functional Ops helper.
    /// </summary>
    public static class TOp
    {

        /// <summary>
        ///  Construct a vector with lanes elements
        ///   where its i-th element equals offset + i * stride.
        ///  This is useful to construct a index for a continuous vector load.
        ///  <remarks>
        ///   NOTE the stride calc by the buffer's Elemtype
        ///   if buffer's Datatype = float32*3, the 1 stride mean skip 1 float32.
        /// </remarks>
        ///  <example>
        ///  - ramp(0, 1, 3) = [0, 1, 2] = [(0 + i * 1) for i in 3]
        ///  - ramp(1, 2, 4) = [1, 3, 5, 7] = [(1 + i * 2) for i in 4]
        /// </example>
        /// </summary>
        /// <param name="offset">The base expression.</param>
        /// <param name="stride">The stride of the ramp.</param>
        /// <param name="lanes">The lanes of the expression.</param>
        public static Call Ramp(Expr offset, Expr stride, int lanes) => new Call(new TIR.Ramp(lanes), offset, stride);

        /// <summary>
        /// Load the **One** value from buffer_var.
        /// Equivalent to ((ElemType*)buffer_var)[index]
        /// <remarks>
        /// If the buffer has packed type like float32*4, but we load the index with lanes 1, so will return only one float32.
        /// </remarks>
        /// <example>
        /// case 1, type = uint32:
        ///   uint32* buffer;
        ///   auto loaded_val = buffer[index]
        /// case 2, type = float32x3
        ///   NOTE the buffer actual type is float32x3, but our index will convert it to float type.
        ///   float32x3 old_buffer;
        ///   float* buffer = static_cast<float*>(old_buffer);
        ///   NOTE then we use Ramp get the index
        ///   index = Ramp(base,1,3)
        ///   auto loaded_val = float32x3(buffer[index.v0], buffer[index.v1], buffer[index.v2]);
        ///                   = float32x3(buffer[base+(0*1)], buffer[base+(1*1)], buffer[base+(2*1)]);
        /// </example>
        /// </summary>
        /// <param name="handle">The buffer handle variable in the load expression.</param>
        /// <param name="index">The index in the load.</param>
        public static Call Load(Var handle, Expr index) => new Call(new Load(), handle, index);

        /// <summary>
        /// Store value to the buffer.
        /// Equivalent to ((DType*)buffer_var)[index] = value.
        /// where DType is the type specified by type().element_of().
        /// <example>
        /// if type = float32x3, then the store will corresponds to
        /// <code>
        ///  auto buffer = static_cast<float*>(buffer_var);
        ///  buffer[index.v0] = value.v0;
        ///  buffer[index.v1] = value.v1;
        ///  buffer[index.v2] = value.v2;
        /// </code>
        /// </example>
        /// </summary>
        /// <param name="handle">The buffer Variable.</param>
        /// <param name="value">The value we want to store.</param>
        /// <param name="index">The index in the store expression.</param>
        /// <returns></returns>
        public static Call Store(Var handle, Expr value, Expr index) => new Call(new Store(), handle, value, index);


        /// <summary>
        /// make a const by value and lanes.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <param name="lanes"></param>
        /// <returns></returns>
        // public static Call MakeConst<T>(T value, Expr lanes)
        // {
        //     return new Call(new MakeConst<T>(value), lanes);
        // }

        /// <summary>
        /// get the current expr's lanes
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        // public static Call LanesOp(Expr input)
        // {
        //     return new Call(new LanesOp(), input);
        // }
    }
}