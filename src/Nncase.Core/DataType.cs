// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Reflection;
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
        [Display(Name = "i8")]
        Int8,

        /// <summary>
        /// Int16.
        /// </summary>
        [Display(Name = "i16")]
        Int16,

        /// <summary>
        /// Int32.
        /// </summary>
        [Display(Name = "i32")]
        Int32,

        /// <summary>
        /// Int64.
        /// </summary>
        [Display(Name = "i64")]
        Int64,

        /// <summary>
        /// UInt8.
        /// </summary>
        [Display(Name = "u8")]
        UInt8,

        /// <summary>
        /// UInt16.
        /// </summary>
        [Display(Name = "u16")]
        UInt16,

        /// <summary>
        /// UInt32.
        /// </summary>
        [Display(Name = "u32")]
        UInt32,

        /// <summary>
        /// UInt64.
        /// </summary>
        [Display(Name = "u64")]
        UInt64,

        /// <summary>
        /// Float16 (Half).
        /// </summary>
        [Display(Name = "f16")]
        Float16,

        /// <summary>
        /// Float32.
        /// </summary>
        [Display(Name = "f32")]
        Float32,

        /// <summary>
        /// Float64.
        /// </summary>
        [Display(Name = "f64")]
        Float64,

        /// <summary>
        /// BFloat16.
        /// </summary>
        [Display(Name = "bf16")]
        BFloat16,

        /// <summary>
        /// Boolean.
        /// </summary>
        [Display(Name = "bool")]
        Bool,

        /// <summary>
        /// String.
        /// </summary>
        [Display(Name = "str")]
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
            { typeof(int).TypeHandle, DataType.Int32 },
            { typeof(float).TypeHandle, DataType.Float32 },
            { typeof(double).TypeHandle, DataType.Float64 }
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

        /// <summary>
        /// Get display name for a datatype.
        /// </summary>
        /// <param name="dataType">Datatype.</param>
        /// <returns>The display name.</returns>
        public static string GetDisplayName(DataType dataType)
        {
            return typeof(DataType).GetField(Enum.GetName(dataType)!)!.GetCustomAttribute<DisplayAttribute>()?.GetName() ?? dataType.ToString();
        }
    }
}
