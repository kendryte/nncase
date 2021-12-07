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
    /// HardMax expression.
    /// </summary>
    public sealed record HardMax() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(HardMax), 0, "input");

        /// <summary>
        /// Gets axis.
        /// </summary>
        public static readonly ParameterInfo Axis = new(typeof(HardMax), 1, "axis", IsIntegralScalar());
        
        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType axis) => input;
    }
}
