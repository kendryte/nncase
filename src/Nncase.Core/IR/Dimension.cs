// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR
{
    /// <summary>
    /// Dimension kind.
    /// </summary>
    public enum DimensionKind
    {
        /// <summary>
        /// Unknown dimension.
        /// </summary>
        Unknown,

        /// <summary>
        /// Fixed dimesnion.
        /// </summary>
        Fixed,
    }

    /// <summary>
    /// Shape dimension.
    /// </summary>
    public struct Dimension
    {
        /// <summary>
        /// An unknown dimension.
        /// </summary>
        public static readonly Dimension Unknown = default;

        /// <summary>
        /// Initializes a new instance of the <see cref="Dimension"/> struct.
        /// </summary>
        /// <param name="value">Dimension value.</param>
        public Dimension(long value)
        {
            Kind = DimensionKind.Fixed;
            Value = value;
        }

        /// <summary>
        /// Gets kind.
        /// </summary>
        public DimensionKind Kind { get; }

        /// <summary>
        /// Gets value.
        /// </summary>
        public long? Value { get; }
    }
}
