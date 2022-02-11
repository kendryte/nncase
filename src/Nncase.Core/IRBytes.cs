// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Nncase.IR
{
    public struct IRBytes : IStructuralEquatable, IEquatable<IRBytes>
    {
        private byte[] _array;

        public IRBytes(byte[] array)
        {
            _array = array;
        }

        /// <summary>
        /// get raw byte data for readonly interface.
        /// </summary>
        /// <returns></returns>
        public ReadOnlySpan<byte> Array => _array;

        /// <summary>
        /// get raw byte data for c/python interface
        /// todo maybe need remove it, because we need pervent modify the raw data.
        /// </summary>
        /// <returns></returns>
        public byte[] Data() => _array;

        public bool Equals(object? other, IEqualityComparer comparer)
        {
            return ((IStructuralEquatable)_array).Equals(other, comparer);
        }

        public override bool Equals(object? obj)
        {
            return obj is IRBytes bytes && Equals(bytes);
        }

        public bool Equals(IRBytes other)
        {
            return StructuralComparisons.StructuralEqualityComparer.Equals(_array, other._array);
        }

        public int GetHashCode(IEqualityComparer comparer)
        {
            return ((IStructuralEquatable)_array).GetHashCode(comparer);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(StructuralComparisons.StructuralEqualityComparer.GetHashCode(_array));
        }

        public static bool operator ==(IRBytes left, IRBytes right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(IRBytes left, IRBytes right)
        {
            return !(left == right);
        }

        public static implicit operator byte[](IRBytes irByte) => irByte._array;

        public static implicit operator ReadOnlySpan<byte>(IRBytes irByte) => irByte._array;

        public static implicit operator IRBytes(byte[] array) => new IRBytes(array);

        public Memory<T> ToMemory<T>()
          where T : unmanaged => Microsoft.Toolkit.HighPerformance.MemoryExtensions.Cast<byte, T>(_array.AsMemory<byte>());
    }
}