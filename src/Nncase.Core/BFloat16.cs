// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase
{
    /// <summary>
    /// BFloat16.
    /// </summary>
    public struct BFloat16 : IEquatable<BFloat16>
    {
        /// <summary>
        /// bfloat16 representation bits.
        /// </summary>
        public ushort value;

        /// <summary>
        /// Initializes a new instance of the <see cref="BFloat16"/> struct.
        /// </summary>
        /// <param name="v">raw value.</param>
        public BFloat16(ushort v)
        {
            value = v;
        }

        /// <summary>
        /// Converts to ushort.
        /// </summary>
        /// <param name="bf">instance of BFloat16.</param>
        /// <returns>value member</returns>
        public static implicit operator ushort(BFloat16 bf) { return bf.value; }

        /// <summary>
        /// Converts a 16-bit unsigned integer to a BFloat16.
        /// </summary>
        /// <param name="value">A 16-bit unsigned integer.</param>
        /// <returns>A BFloat16 that represents the converted 16-bit unsigned integer.</returns>
        public static implicit operator BFloat16(ushort value) { return new BFloat16(value); }

        /// <summary>
        /// Compares values of two BFloat16 for binary equality.
        /// </summary>
        /// <param name="lhs">lhs.</param>
        /// <param name="rhs">rhls.</param>
        /// <returns>result of value comparisons</returns>
        public static bool operator ==(BFloat16 lhs, BFloat16 rhs) { return lhs.value == rhs.value; }

        /// <summary>
        /// Compares values of two BFloat16 for binary inequality.
        /// </summary>
        /// <param name="lhs">lhs.</param>
        /// <param name="rhs">rhls.</param>
        /// <returns>result of value comparisons</returns>
        public static bool operator !=(BFloat16 lhs, BFloat16 rhs) { return lhs.value != rhs.value; }

        public static BFloat16 RoundToBFloat16(float value)
        {
            if (float.IsNaN(value))
            {
                // If the value is a NaN, squash it to a qNaN with msb of fraction set,
                // this makes sure after truncation we don't end up with an inf.
                //
                // qNaN magic: All exponent bits set + most significant bit of fraction
                // set.
                return new BFloat16(0x7fc0);
            }

            var input = Unsafe.As<float, uint>(ref value);
            // Least significant bit of resulting bfloat.
            uint lsb = (input >> 16) & 1;
            uint roundingBias = 0x7fff + lsb;
            input += roundingBias;
            return new BFloat16((ushort) (input >> 16));
        }

        /// <summary>
        /// Returns a value indicating whether this instance and other BFloat16 represent the same value.
        /// </summary>
        /// <param name="other">A BFloat16 object to compare to this instance.</param>
        /// <returns>true if other.value is equal to this instance; otherwise, false.</returns>
        public bool Equals(BFloat16 other)
        {
            return (other == this);
        }

        public static implicit operator float(BFloat16 input)
        {
            float value;
            Unsafe.SkipInit(out value);
            Unsafe.As<float, int>(ref value) = input.value << 16;
            return value;
        }
        
        /// <summary>
        /// Returns a value indicating whether this instance and a specified System.Object
        /// represent the same type and value.
        /// </summary>
        /// <param name="obj">An System.Object.</param>
        /// <returns>true if obj is BFloat16 its value is equal to this instance; otherwise, false.</returns>
        public override bool Equals(object? obj)
        {
            bool result = false;
            if (obj is BFloat16)
            {
                BFloat16 bfl16 = (BFloat16)obj;
                result = (bfl16 == this);
            }
            return result;
        }

        /// <summary>
        /// Returns the hash code for this instance.
        /// </summary>
        /// <returns>A 32-bit signed integer hash code.</returns>
        public override int GetHashCode()
        {
            return value.GetHashCode();
        }
    }
}
