// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.Utility;

namespace Nncase.IR.Tensors
{
    /// <summary>
    /// CumSum expression.
    /// </summary>
    public sealed record CumSum() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(CumSum), 0, "input");

        /// <summary>
        /// Gets axis.
        /// </summary>
        public static readonly ParameterInfo Axis = new(typeof(CumSum), 1, "axis");

        /// <summary>
        /// Gets exclusive.
        /// </summary>
        public static readonly ParameterInfo Exclusive = new(typeof(CumSum), 2, "exclusive", IsBoolScalar());

        /// <summary>
        /// Gets reverse.
        /// </summary>
        public static readonly ParameterInfo Reverse = new(typeof(CumSum), 3, "reverse", IsBoolScalar());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType axis,
            TensorType exclusive, TensorType reverse) => input;
    }
}
