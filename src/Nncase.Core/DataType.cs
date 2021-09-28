// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase
{
    /// <summary>
    /// Data type.
    /// </summary>
    public enum DataType : byte
    {
        /// <summary>
        /// Int8.
        /// </summary>
        Int8,

        /// <summary>
        /// Int16.
        /// </summary>
        Int16,

        /// <summary>
        /// Int32.
        /// </summary>
        Int32,

        /// <summary>
        /// Int64.
        /// </summary>
        Int64,

        /// <summary>
        /// UInt8.
        /// </summary>
        UInt8,

        /// <summary>
        /// UInt16.
        /// </summary>
        UInt16,

        /// <summary>
        /// UInt32.
        /// </summary>
        UInt32,

        /// <summary>
        /// UInt64.
        /// </summary>
        UInt64,

        /// <summary>
        /// Float16 (Half).
        /// </summary>
        Float16,

        /// <summary>
        /// Float32.
        /// </summary>
        Float32,

        /// <summary>
        /// Float64.
        /// </summary>
        Float64,

        /// <summary>
        /// BFloat16.
        /// </summary>
        BFloat16,

        /// <summary>
        /// Boolean.
        /// </summary>
        Bool,

        /// <summary>
        /// String.
        /// </summary>
        String,
    }

    /// <summary>
    /// Data type helper.
    /// </summary>
    public static class DataTypes
    {
        private static readonly Dictionary<RuntimeTypeHandle, DataType> _typeToDataTypes = new()
        {
            { typeof(byte).TypeHandle, DataType.UInt8 },
        };

        /// <summary>
        /// Get data type from CLR type.
        /// </summary>
        /// <param name="t">CLR type.</param>
        /// <returns>Data type.</returns>
        public static DataType FromType(Type t)
        {
            if (_typeToDataTypes.TryGetValue(t.TypeHandle, out var dataType))
            {
                return dataType;
            }

            throw new ArgumentOutOfRangeException("Unsupported CLR type: " + t.FullName);
        }

        /// <summary>
        /// Get data type from CLR type.
        /// </summary>
        /// <typeparam name="T">CLR type.</typeparam>
        /// <returns>Data type.</returns>
        public static DataType FromType<T>()
            where T : unmanaged
            => FromType(typeof(T));

        /// <summary>
        /// Convert unmanaged type to bytes.
        /// </summary>
        /// <typeparam name="T">Unmanaged type.</typeparam>
        /// <param name="value">Value.</param>
        /// <returns>Converted bytes.</returns>
        public static byte[] GetBytes<T>(T value)
            where T : unmanaged
            => MemoryMarshal.AsBytes(MemoryMarshal.CreateSpan(ref value, 1)).ToArray();

        /// <summary>
        /// Convert span of unmanaged type to bytes.
        /// </summary>
        /// <typeparam name="T">Unmanaged type.</typeparam>
        /// <param name="span">Span.</param>
        /// <returns>Converted bytes.</returns>
        public static byte[] GetBytes<T>(ReadOnlySpan<T> span)
            where T : unmanaged
            => MemoryMarshal.AsBytes(span).ToArray();
    }
}
