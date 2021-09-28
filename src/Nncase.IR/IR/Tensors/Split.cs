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
    /// Split expression.
    /// </summary>
    public record Split() : Op(ImmutableArray.Create(
        new ParameterInfo("input"),
        new ParameterInfo("axis"),
        new ParameterInfo("sections")))
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
        /// Gets sections.
        /// </summary>
        public ParameterInfo Sections => Parameters[2];

        /// <inheritdoc/>
        public override Type InferInvokeResultType(ITypeInferenceContext context)
        {
            throw new NotImplementedException();
        }
    }
}
