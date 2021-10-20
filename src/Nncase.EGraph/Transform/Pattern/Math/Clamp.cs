// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Transform.Pattern.Math
{
    /// <summary>
    /// Clamp expression.
    /// </summary>
    public record ClampPattern() : OpPattern(ImmutableArray.Create(
        new ParameterInfoPattern("input"), new ParameterInfoPattern("min"), new ParameterInfoPattern("max")))
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public ParameterInfoPattern Input => Parameters[0];

        /// <summary>
        /// Gets min.
        /// </summary>
        public ParameterInfoPattern Min => Parameters[1];

        /// <summary>
        /// Gets max.
        /// </summary>
        public ParameterInfoPattern Max => Parameters[2];

    }
}
