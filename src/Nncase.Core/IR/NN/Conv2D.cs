// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Pattern;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.NN
{
    /// <summary>
    /// Conv2D.
    /// </summary>
    [PatternFunctionalGenerator]
    public sealed record Conv2D(PadMode PadMode) : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(Conv2D), 0, "input", IsRank(4));

        /// <summary>
        /// Gets Weights.
        /// </summary>
        public static readonly ParameterInfo Weights = new(typeof(Conv2D), 1, "weights", IsRank(4));

        /// <summary>
        /// Gets Bias.
        /// </summary>
        public static readonly ParameterInfo Bias = new(typeof(Conv2D), 2, "bias", IsRank(1));

        /// <summary>
        /// Gets Stride.
        /// </summary>
        public static readonly ParameterInfo Stride = new(typeof(Conv2D), 3, "stride", IsRank(1) & IsIntegral());

        /// <summary>
        /// Gets Padding.
        /// </summary>
        public static readonly ParameterInfo Padding = new(typeof(Conv2D), 4, "padding", IsRank(2) & IsIntegral());

        /// <summary>
        /// Gets Dilation.
        /// </summary>
        public static readonly ParameterInfo Dilation = new(typeof(Conv2D), 5, "dilation", IsRank(1) & IsIntegral());

        /// <summary>
        /// Gets Groups.
        /// </summary>
        public static readonly ParameterInfo Groups = new(typeof(Conv2D), 6, "groups", IsScalar() & IsIntegral());
    }
}
