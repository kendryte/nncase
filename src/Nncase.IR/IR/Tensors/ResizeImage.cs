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
    /// ResizeImage expression.
    /// </summary>
    public sealed record ResizeImage(ImageResizeMode ResizeMode) : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(ResizeImage), 0, "input");

        /// <summary>
        /// Gets new_size.
        /// </summary>
        public static readonly ParameterInfo NewSize = new(typeof(ResizeImage), 1, "new_size");
        
        /// <summary>
        /// Gets AlignCorners.
        /// </summary>
        public static readonly ParameterInfo AlignCorners = new(typeof(ResizeImage), 2, "align_corners");
        
        /// <summary>
        /// Gets HalfPixelCenters.
        /// </summary>
        public static readonly ParameterInfo HalfPixelCenters = new(typeof(ResizeImage), 3, "half_pixel_centers");

        /// <inheritdoc/>
        public override IRType InferInvokeResultType(ITypeInferenceContext context)
        {
            throw new NotImplementedException();
        }
    }
}
