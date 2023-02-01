// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// Value range.
/// </summary>
/// <typeparam name="T">Value type.</typeparam>
public record struct ValueRange<T>(T Min, T Max)
    where T : unmanaged, IEquatable<T>, IComparable<T>
{
    /// <summary>
    /// Gets full value range.
    /// </summary>
    public static ValueRange<T> Full { get; } = (Limits.MinValue, Limits.MaxValue);

    public bool IsFull => this == Full;

    /// <summary>
    /// Convert 2 elements tuple to <see cref="ValueRange{T}"/>.
    /// </summary>
    /// <param name="tuple">Tuple.</param>
    public static implicit operator ValueRange<T>((T Min, T Max) tuple) =>
        new ValueRange<T> { Min = tuple.Min, Max = tuple.Max };

    public ValueRange<T> Union(ValueRange<T> range)
    {
        var min = Min.CompareTo(range.Min) < 0 ? Min : range.Min;
        var max = Max.CompareTo(range.Max) > 0 ? Max : range.Max;
        return (min, max);
    }

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

                if (typeof(T) == typeof(BFloat16))
                {
                    return (T)(object)BFloat16.NegInfinity;
                }

                if (typeof(T) == typeof(Half))
                {
                    return (T)(object)Half.NegativeInfinity;
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

                if (typeof(T) == typeof(BFloat16))
                {
                    return (T)(object)BFloat16.Infinity;
                }

                if (typeof(T) == typeof(Half))
                {
                    return (T)(object)Half.PositiveInfinity;
                }

                throw new NotSupportedException();
            }
        }
    }
}
