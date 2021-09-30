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
    /// Clamp expression.
    /// </summary>
    public record Clamp() : Op(ImmutableArray.Create(
        new ParameterInfo("input"), new ParameterInfo("min"), new ParameterInfo("max")))
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public ParameterInfo Input => Parameters[0];

        /// <summary>
        /// Gets min.
        /// </summary>
        public ParameterInfo Min => Parameters[1];

        /// <summary>
        /// Gets max.
        /// </summary>
        public ParameterInfo Max => Parameters[2];

        /// <inheritdoc/>
        public override IRType InferInvokeResultType(ITypeInferenceContext context)
        {
            var inputType = context.CheckArgumentType<TensorType>(this, Input);
            var minType = context.CheckArgumentType<TensorType>(this, Min);
            var maxType = context.CheckArgumentType<TensorType>(this, Max);
            return TypeInference.BroadcastType(inputType, minType, maxType).ThrowIfTypeInferenceInterrupt();
        }
    }
}
