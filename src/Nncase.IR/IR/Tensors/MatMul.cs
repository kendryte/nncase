// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR.Tensors
{
    /// <summary>
    /// MatMul expression.
    /// </summary>
    public sealed record MatMul() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(MatMul), 0, "input");

        /// <summary>
        /// Gets Other.
        /// </summary>
        public static readonly ParameterInfo Other = new(typeof(MatMul), 1, "other");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType other)
        {
            if (input.Shape.Rank != 2)
            {
                return new InvalidType("MatMul input_a shape rank is not 2");
            }

            if (other.Shape.Rank != 2)
            {
                return new InvalidType("MatMul input_b shape rank is not 2");
            }

            if (input.Shape[1].IsUnknown || other.Shape[0].IsUnknown)
            {
                return new InvalidType("MatMul input a or b shape is unknown");
            }

            if (input.Shape[1] != other.Shape[0])
            {
                return new InvalidType("MatMul input_a shape[1] != input_b shape[0]");
            }

            return new TensorType(input.DType, new[] { input.Shape[0], other.Shape[1] });
        }
    }
}
