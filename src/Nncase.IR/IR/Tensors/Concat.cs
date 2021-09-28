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
    /// Concat expression.
    /// </summary>
    public record Concat() : Op(ImmutableArray.Create(new ParameterInfo("input"), new ParameterInfo("axis")))
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public ParameterInfo Input => Parameters[0];

        /// <summary>
        /// Gets axis.
        /// </summary>
        public ParameterInfo Axis => Parameters[1];

        /// <inheritdoc/>
        public override Type InferInvokeResultType(ITypeInferenceContext context)
        {
            throw new NotImplementedException();
        }
    }
}
