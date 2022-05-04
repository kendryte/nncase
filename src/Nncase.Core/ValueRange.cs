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
public struct ValueRange<T> : IEquatable<ValueRange<T>>
    where T : unmanaged, IEquatable<T>
{
    /// <summary>
    /// Gets full value range.
    /// </summary>
    public static ValueRange<T> Full => (Limits.MinValue, Limits.MaxValue);
    
    public bool IsFull => this == Full;
    
    /// <summary>
    /// Gets or sets min value.
    /// </summary>
    public T Min { get; set; }

    /// <summary>
    /// Gets or sets max value.
    /// </summary>
    public T Max { get; set; }

    /// <inheritdoc/>
    public bool Equals(ValueRange<T> other)
    {
        return Min.Equals(other.Min) && Max.Equals(other.Max);
    }

    /// <summary>
    /// Convert 2 elements tuple to <see cref="ValueRange{T}"/>.
    /// </summary>
    /// <param name="tuple">Tuple.</param>
    public static implicit operator ValueRange<T>((T Min, T Max) tuple) =>
        new ValueRange<T> { Min = tuple.Min, Max = tuple.Max };

    /// <inheritdoc/>
    public static bool operator ==(ValueRange<T>? lhs, ValueRange<T>? rhs)
    {
        if (lhs is null)
        {
            if (rhs is null)
            {
                return true;
            }

            // Only the left side is null.
            return false;
        }
        // Equals handles case of null on right side.
        return lhs.Equals(rhs);
    }

    /// <inheritdoc/>
    public static bool operator !=(ValueRange<T>? lhs, ValueRange<T>? rhs) => !(lhs == rhs);
    
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
                    return (T)(object)(BFloat16.NegInfinity);
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
                    return (T)(object)(BFloat16.Infinity);
                }
                
                throw new NotSupportedException();
            }
        }
    }

    /// <inheritdoc/>
    public override bool Equals(object obj)
    {
        return obj is ValueRange<T> v && this.Equals(v);
    }
}
