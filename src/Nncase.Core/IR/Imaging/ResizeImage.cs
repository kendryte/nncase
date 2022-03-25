// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Imaging
{
    /// <summary>
    /// ResizeImage expression.
    /// </summary>
    public sealed record ResizeImage(
        ImageResizeMode ResizeMode, 
        ImageResizeTransformationMode TransformationMode,
        ImageResizeNearestMode NearestMode, bool IsTFResize = false) : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(ResizeImage), 0, "input", HasRank(r => r >= 2, "RanK >= 2"));

        /// <summary>
        /// Gets roi.
        /// </summary>
        public static readonly ParameterInfo Roi = new(typeof(ResizeImage), 1, "roi", IsFloatScalar());

        /// <summary>
        /// Gets new_size.
        /// </summary>
        public static readonly ParameterInfo NewSize = new(typeof(ResizeImage), 2, "new_size", HasRank(1));
        
        /// <summary>
        /// Gets CubicCoeffA.
        /// </summary>
        public static readonly ParameterInfo CubicCoeffA = new(typeof(ResizeImage), 3, "cubic_coeff_a", IsFloatScalar());
        
        /// <summary>
        /// Gets ExcludeOutside.
        /// </summary>
        public static readonly ParameterInfo ExcludeOutside = new(typeof(ResizeImage), 4, "exclude_outside", IsIntegralScalar());
        
        /// <summary>
        /// Gets ExtrapolationValue.
        /// </summary>
        public static readonly ParameterInfo ExtrapolationValue = new(typeof(ResizeImage), 5, "extrapolation_value", IsFloatScalar());
        
        public static readonly ParameterInfo AlignCorners = new(typeof(ResizeImage), 6, "align_corners");
        public static readonly ParameterInfo HalfPixelCenters = new(typeof(ResizeImage), 7, "half_pixel_centers");
    }
}
