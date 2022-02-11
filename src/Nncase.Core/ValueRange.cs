// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase
{
    /// <summary>
    /// Value range.
    /// </summary>
    /// <typeparam name="T">Value type.</typeparam>
    public struct ValueRange<T>
        where T : unmanaged
    {
        /// <summary>
        /// Gets full value range.
        /// </summary>
        public static ValueRange<T> Full => (Limits.MinValue, Limits.MaxValue);

        /// <summary>
        /// Gets or sets min value.
        /// </summary>
        public T Min { get; set; }

        /// <summary>
        /// Gets or sets max value.
        /// </summary>
        public T Max { get; set; }

        /// <summary>
        /// Convert 2 elements tuple to <see cref="ValueRange{T}"/>.
        /// </summary>
        /// <param name="tuple">Tuple.</param>
        public static implicit operator ValueRange<T>((T Min, T Max) tuple) =>
            new ValueRange<T> { Min = tuple.Min, Max = tuple.Max };

        private static class Limits
        {
            public static T MinValue
            {
                get
                {
                    if (typeof(T) == typeof(byte))
                    {
                        return (T)(object)byte.MinValue;
                    }

                    if (typeof(T) == typeof(float))
                    {
                        return (T)(object)float.NegativeInfinity;
                    }

                    throw new NotSupportedException();
                }
            }

            public static T MaxValue
            {
                get
                {
                    if (typeof(T) == typeof(byte))
                    {
                        return (T)(object)byte.MaxValue;
                    }

                    if (typeof(T) == typeof(float))
                    {
                        return (T)(object)float.PositiveInfinity;
                    }

                    throw new NotSupportedException();
                }
            }
        }
    }
}
