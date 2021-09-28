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
    public record Binary(BinaryOp BinaryOp) : Op(ImmutableArray.Create(new ParameterInfo("lhs"), new ParameterInfo("rhs")))
    {
        /// <summary>
        /// Gets lhs.
        /// </summary>
        public ParameterInfo Lhs => Parameters[0];

        /// <summary>
        /// Gets rhs.
        /// </summary>
        public ParameterInfo Rhs => Parameters[1];

        /// <inheritdoc/>
        public override Type InferInvokeResultType(ITypeInferenceContext context)
        {
            var lhsType = context.CheckArgumentType<TensorType>(this, Lhs);
            var rhsType = context.CheckArgumentType<TensorType>(this, Rhs);
            return TypeInference.BroadcastType(lhsType, rhsType).ThrowIfTypeInferenceInterrupt();
        }
    }
}
