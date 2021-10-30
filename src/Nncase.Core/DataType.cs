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

        private static readonly Dictionary<DataType, int> _DataTypeToLengths = new()
        {
            { DataType.UInt8, 1 },
            { DataType.UInt16, 2 },
            { DataType.UInt32, 4 },
            { DataType.UInt64, 8 },
            { DataType.Int8, 1 },
            { DataType.Int16, 2 },
            { DataType.Int32, 4 },
            { DataType.Int64, 8 },
            { DataType.Float16, 2 },
            { DataType.BFloat16, 2 },
            { DataType.Float32, 4 },
            { DataType.Float64, 8 }
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

        public static int GetLength(DataType dataType) => _DataTypeToLengths[dataType];

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

        public static T ToScalar<T>(DataType dataType, byte[] bytes, int start = 0)
        where T : unmanaged
        => dataType switch
        {
            DataType.Int64 => (T)(object)BitConverter.ToInt64(bytes, start),
            DataType.Int32 => (T)(object)BitConverter.ToInt32(bytes, start),
            DataType.Int16 => (T)(object)BitConverter.ToInt16(bytes, start),
            DataType.Int8 => (T)(object)bytes[start],
            DataType.UInt64 => (T)(object)BitConverter.ToUInt64(bytes, start),
            DataType.UInt32 => (T)(object)BitConverter.ToUInt32(bytes, start),
            DataType.UInt16 => (T)(object)BitConverter.ToUInt16(bytes, start),
            DataType.UInt8 => (T)(object)bytes[start],
            DataType.Float64 => (T)(object)BitConverter.ToDouble(bytes, start),
            DataType.Float32 => (T)(object)BitConverter.ToSingle(bytes, start),
            DataType.BFloat16 => (T)(object)(new BFloat16(bytes[start])),
            DataType.Float16 => (T)(object)(Half)(bytes[start]),
            DataType.Bool => (T)(object)BitConverter.ToBoolean(bytes, start),
            // DataType.String => (T)(object)BitConverter.ToString(bytes, start),
            _ => throw new InvalidCastException($"Can't Cast the {dataType.ToString()}!")
        };
    }
}
