// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR.NN
{
    /// <summary>
    /// Sigmoid expression.
    /// </summary>
    public sealed record Sigmoid() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(Sigmoid), 0, "input");

        /// <inheritdoc/>
        public override IRType InferInvokeResultType(ITypeInferenceContext context)
        {
            var inputType = context.CheckArgumentType<TensorType>(this, Input);
            return inputType;
        }
    }
}
