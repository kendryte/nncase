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
        /// </summary>
        /// <param name="load_dtype">load data type</param>
        /// <param name="buffer_handle">The buffer handle variable in the load expression.</param>
        /// <param name="index">The index in the load.</param>
        /// <param name="predicate">The load predicate.</param>
        public static Call Load(DataType load_dtype, Var buffer_handle, Expr index, Expr? predicate)
        {
            int lanes = 1;
            if (buffer_handle.TypeAnnotation is PointerType ptype)
            {
                lanes = ptype.DType.Lanes;
            }
            return new Call(new Load(load_dtype), buffer_handle, index, predicate ?? Const.FromScalar(1, lanes));
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
    }
}