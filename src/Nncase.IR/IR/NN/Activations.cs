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
    public record Sigmoid() : Op(ImmutableArray.Create(new ParameterInfo("input")))
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public ParameterInfo Input => Parameters[0];

        /// <inheritdoc/>
        public override IRType InferInvokeResultType(ITypeInferenceContext context)
        {
            var inputType = context.CheckArgumentType<TensorType>(this, Input);
            return inputType;
        }
    }
}
