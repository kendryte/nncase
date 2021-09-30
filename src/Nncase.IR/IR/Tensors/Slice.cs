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
    /// Slice expression.
    /// </summary>
    public record Slice() : Op(ImmutableArray.Create(
        new ParameterInfo("input"),
        new ParameterInfo("begins"),
        new ParameterInfo("ends"),
        new ParameterInfo("axes"),
        new ParameterInfo("strides")))
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public ParameterInfo Input => Parameters[0];

        /// <summary>
        /// Gets begins.
        /// </summary>
        public ParameterInfo Begins => Parameters[1];

        /// <summary>
        /// Gets ends.
        /// </summary>
        public ParameterInfo Ends => Parameters[2];

        /// <summary>
        /// Gets axes.
        /// </summary>
        public ParameterInfo Axes => Parameters[3];

        /// <summary>
        /// Gets strides.
        /// </summary>
        public ParameterInfo Strides => Parameters[4];

        /// <inheritdoc/>
        public override IRType InferInvokeResultType(ITypeInferenceContext context)
        {
            throw new NotImplementedException();
        }
    }
}
