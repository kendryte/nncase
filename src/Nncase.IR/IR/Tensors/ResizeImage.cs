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
            var out_shape = input.Shape.ToArray();
            if (context.GetArgument(this, NewSize) is Const new_size_con)
            {
                var ts_new_size = new_size_con.ToTensor<int>();
                switch (out_shape.Length)
                {
                    case 2 or 3:
                        out_shape[0] = ts_new_size[0];
                        out_shape[1] = ts_new_size[1];
                        break;
                    case > 3:
                        out_shape[^3] = ts_new_size[0];
                        out_shape[^2] = ts_new_size[1];
                        break;
                }
            }
            else
            {

                switch (out_shape.Length)
                {
                    case 2 or 3:
                        out_shape[0] = Dimension.Unknown;
                        out_shape[1] = Dimension.Unknown;
                        break;
                    case > 3:
                        out_shape[^3] = Dimension.Unknown;
                        out_shape[^2] = Dimension.Unknown;
                        break;
                }
            }
            return input with { Shape = new Shape(out_shape) };
        }
    }
}
