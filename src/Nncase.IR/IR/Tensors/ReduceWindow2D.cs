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
    /// ReduceWindow2D.
    /// </summary>
    public sealed record ReduceWindow2D(ReduceOp ReduceOp) : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(ReduceWindow2D), 0, "input");
        
        /// <summary>
        /// Get init_value.
        /// </summary>
        public static readonly ParameterInfo InitValue = new(typeof(Reduce), 1, "init_value");
        
        /// <summary>
        /// Get filter.
        /// </summary>
        public static readonly ParameterInfo Filter = new(typeof(Reduce), 2, "filter");
        
        /// <summary>
        /// Gets Stride.
        /// </summary>
        public static readonly ParameterInfo Stride = new(typeof(ReduceWindow2D), 3, "stride");

        /// <summary>
        /// Gets Padding.
        /// </summary>
        public static readonly ParameterInfo Padding = new(typeof(ReduceWindow2D), 4, "padding");

        /// <summary>
        /// Gets Dilation.
        /// </summary>
        public static readonly ParameterInfo Dilation = new(typeof(ReduceWindow2D), 5, "dilation");
        
        /// <inheritdoc/>
        public override IRType InferInvokeResultType(ITypeInferenceContext context)
        {
            throw new NotImplementedException();
        }
    }
}