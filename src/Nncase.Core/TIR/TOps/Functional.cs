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
    public static class TOps
    {
        /// <summary>
        /// Load the value from buffer_var.
        /// Equivalent to ((DType*)buffer_var)[index]
        /// where DType is the type specified by type().element_of().
        /// <example>
        /// if type = float32x3, then the load will corresponds to
        ///   auto buffer = static_cast<float*>(buffer_var);
        ///   auto loaded_val = float32x3(buffer[index.v0], buffer[index.v1], buffer[index.v2]);
        /// </example>
        /// TODO add load with different type
        /// </summary>
        /// <param name="buffer_handle">The buffer handle variable in the load expression.</param>
        /// <param name="index">The index in the load.</param>
        /// <param name="predicate">The load predicate.</param>
        public static Call Load(Var buffer_handle, Expr index, Expr? predicate = null)
        {

            return new Call(new Load(), buffer_handle, index, predicate ?? MakeConst<int>(1, LanesOp(buffer_handle)));
        }

        /// <summary>
        ///  Construct a vector with lanes elements
        ///        where its i-th element equals base + i * stride.
        ///  This is useful to construct a index for a continuous vector load.
        ///
        ///  Examples:
        ///  - ramp(0, 1, 3) = [0, 1, 2]
        ///  - ramp(1, 2, 4) = [1, 3, 5, 7]
        /// 
        /// </summary>
        /// <param name="base_offset"></param>
        /// <param name="stride"></param>
        /// <param name="lanes"></param>
        /// <returns></returns>
        public static Call Ramp(Expr base_offset, Expr stride, int lanes)
        {
            return new Call(new Ramp(lanes), base_offset, stride);
        }

        /// <summary>
        /// make a const by value and lanes.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <param name="lanes"></param>
        /// <returns></returns>
        public static Call MakeConst<T>(T value, Expr lanes)
        {
            return new Call(new MakeConst<T>(value), lanes);
        }

        /// <summary>
        /// get the current expr's lanes
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public static Call LanesOp(Expr input)
        {
            return new Call(new LanesOp(), input);
        }
    }
}