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
    /// Conv2D.
    /// </summary>
    public sealed record Conv2D(PadMode padMode) : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(Conv2D), 0, "input");

        /// <summary>
        /// Gets Weights.
        /// </summary>
        public static readonly ParameterInfo Weights = new(typeof(Conv2D), 1, "weights");

        /// <summary>
        /// Gets Bias.
        /// </summary>
        public static readonly ParameterInfo Bias = new(typeof(Conv2D), 2, "bias");

        /// <summary>
        /// Gets Stride.
        /// </summary>
        public static readonly ParameterInfo Stride = new(typeof(Conv2D), 3, "stride");

        /// <summary>
        /// Gets Padding.
        /// </summary>
        public static readonly ParameterInfo Padding = new(typeof(Conv2D), 4, "padding");

        /// <summary>
        /// Gets Dilation.
        /// </summary>
        public static readonly ParameterInfo Dilation = new(typeof(Conv2D), 5, "dilation");
        
        /// <summary>
        /// Gets Groups.
        /// </summary>
        public static readonly ParameterInfo Groups = new(typeof(Conv2D), 6, "groups");
        
        /// <inheritdoc/>
        public override IRType InferInvokeResultType(ITypeInferenceContext context)
        {
            throw new NotImplementedException();
        }
    }
}
