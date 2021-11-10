// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR.Math
{
    /// <summary>
    /// Binary expression.
    /// </summary>
    public sealed record Binary(BinaryOp BinaryOp) : Op
    {
        /// <summary>
        /// Gets lhs.
        /// </summary>
        public static readonly ParameterInfo Lhs = new(typeof(Binary), 0, "lhs");

        /// <summary>
        /// Gets rhs.
        /// </summary>
        public static readonly ParameterInfo Rhs = new(typeof(Binary), 1, "rhs");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType lhs, TensorType rhs)
        {
            return TypeInference.BroadcastType(lhs, rhs);
        }
    }
}
