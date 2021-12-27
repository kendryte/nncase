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
    /// Expand expression.
    /// </summary>
    public sealed record Expand() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(Expand), 0, "input");

        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Shape = new(typeof(Expand), 1, "shape");
        
        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType shape)
        {
            if (context.GetArgument(this, Shape) is Const constShape)
            {
                return new TensorType(input.DType, constShape.ToArray<int>());
            }
            else
            {
                return new InvalidType("Expand Shape need const value");
            }
        }
    }
}
