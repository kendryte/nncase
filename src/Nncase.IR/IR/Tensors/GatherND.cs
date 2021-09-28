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
    /// GatherND expression.
    /// </summary>
    public record GatherND() : Op(ImmutableArray.Create(
        new ParameterInfo("input"), new ParameterInfo("axis"), new ParameterInfo("batch_dims"), new ParameterInfo("index")))
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public ParameterInfo Input => Parameters[0];

        /// <summary>
        /// Gets axis.
        /// </summary>
        public ParameterInfo Axis => Parameters[1];

        /// <summary>
        /// Gets batch dims.
        /// </summary>
        public ParameterInfo BatchDims => Parameters[2];

        /// <summary>
        /// Gets index.
        /// </summary>
        public ParameterInfo Index => Parameters[3];

        /// <inheritdoc/>
        public override Type InferInvokeResultType(ITypeInferenceContext context)
        {
            throw new NotImplementedException();
        }
    }
}
