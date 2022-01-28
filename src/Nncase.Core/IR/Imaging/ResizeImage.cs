// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.Utility;

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
        public static readonly ParameterInfo Input = new(typeof(ResizeImage), 0, "input", HasRank(r => r >= 2, "RanK >= 2"));

        /// <summary>
        /// Gets new_size.
        /// </summary>
        public static readonly ParameterInfo NewSize = new(typeof(ResizeImage), 1, "new_size", HasRank(1));

        /// <summary>
        /// Gets AlignCorners.
        /// </summary>
        public static readonly ParameterInfo AlignCorners = new(typeof(ResizeImage), 2, "align_corners", IsScalar() & IsIntegral());

        /// <summary>
        /// Gets HalfPixelCenters.
        /// </summary>
        public static readonly ParameterInfo HalfPixelCenters = new(typeof(ResizeImage), 3, "half_pixel_centers", IsScalar() & IsIntegral());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context,
           TensorType input, TensorType new_size,
           TensorType align_corners, TensorType half_pixel_centers)
        {
            var newSize = context.GetArgument(this, NewSize);
            return TypeInference.ResizeType(input, newSize);
        }
    }
}
