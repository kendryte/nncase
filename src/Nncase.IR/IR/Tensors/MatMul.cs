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
        public IRType InferInvokeResultType(ITypeInferenceContext context)
        {
            throw new NotImplementedException();
        }
    }
}
