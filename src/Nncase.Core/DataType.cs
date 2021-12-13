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
    /// Single Elem type.
    /// </summary>
    public enum ElemType : byte
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

        /// <summary>
        /// Invalid.
        /// </summary>
        [Display(Name = "Invalid")]
        Invalid,
    }

    /// <summary>
    /// The storge data Type, for simd/npu/gpu we need support packed ElemType
    /// <example>
    /// float32*4
    /// int8*2
    /// </example>
    /// </summary>
    /// <param name="ElemType"></param>
    /// <param name="Lanes"></param>
    public sealed record DataType(ElemType ElemType, int Lanes = 1)
    {
        public static implicit operator DataType(ElemType ElemType) => new DataType(ElemType, 1);
        public static DataType Int8 => ElemType.Int8;
        public static DataType Int16 => ElemType.Int16;
        public static DataType Int32 => ElemType.Int32;
        public static DataType Int64 => ElemType.Int64;
        public static DataType UInt8 => ElemType.UInt8;
        public static DataType UInt16 => ElemType.UInt16;
        public static DataType UInt32 => ElemType.UInt32;
        public static DataType UInt64 => ElemType.UInt64;
        public static DataType Float16 => ElemType.Float16;
        public static DataType Float32 => ElemType.Float32;
        public static DataType Float64 => ElemType.Float64;
        public static DataType BFloat16 => ElemType.BFloat16;
        public static DataType Bool => ElemType.Bool;
        public static DataType String => ElemType.String;
        public static DataType Invalid => ElemType.Invalid;

        /// <summary>
        /// check current compatible with other datatype
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public Compatible CompatibleWith(DataType other) => new Compatible((this.ElemType == other.ElemType && this.Lanes % other.Lanes == 0), $"this {this} != other {other}");
    }

    public sealed record Compatible(bool IsCompatible, string Reason)
    {
        public static implicit operator bool(Compatible compatible) => compatible.IsCompatible;
    }

    /// <summary>
    /// Data type helper.
    /// </summary>
    public static class DataTypes
    {
        private static readonly Dictionary<RuntimeTypeHandle, DataType> _typeToDataTypes = new()
        {
            { typeof(bool).TypeHandle, DataType.Bool },
            { typeof(sbyte).TypeHandle, DataType.Int8 },
            { typeof(byte).TypeHandle, DataType.UInt8 },
            { typeof(int).TypeHandle, DataType.Int32 },
            { typeof(uint).TypeHandle, DataType.UInt32 },
            { typeof(long).TypeHandle, DataType.Int64 },
            { typeof(ulong).TypeHandle, DataType.UInt64 },
            { typeof(float).TypeHandle, DataType.Float32 },
            { typeof(double).TypeHandle, DataType.Float64 },
            { typeof(char).TypeHandle, DataType.String },
        };

        private static readonly Dictionary<ElemType, int> _ElemTypeToLengths = new()
        {
            { ElemType.Bool, 1 },
            { ElemType.UInt8, 1 },
            { ElemType.UInt16, 2 },
            { ElemType.UInt32, 4 },
            { ElemType.UInt64, 8 },
            { ElemType.Int8, 1 },
            { ElemType.Int16, 2 },
            { ElemType.Int32, 4 },
            { ElemType.Int64, 8 },
            { ElemType.Float16, 2 },
            { ElemType.BFloat16, 2 },
            { ElemType.Float32, 4 },
            { ElemType.Float64, 8 }
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
        /// get data type total bytes lengths.
        /// <example>
        /// GetLength(float32) => 4
        /// </example>
        /// </summary>
        /// <param name="dataType"></param>
        /// <returns></returns>
        public static int GetLength(DataType dataType) => _ElemTypeToLengths[dataType.ElemType] * dataType.Lanes;

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
            return typeof(ElemType).GetField(Enum.GetName(dataType.ElemType)!)!.GetCustomAttribute<DisplayAttribute>()?.GetName() ?? dataType.ToString();
        }

        public static T ToScalar<T>(DataType srcType, byte[] bytes, int start = 0)
          where T : unmanaged
          => srcType switch
          {
              { ElemType: ElemType.Int64, Lanes: 1 } => (T)(object)BitConverter.ToInt64(bytes, start),
              { ElemType: ElemType.Int32, Lanes: 1 } => (T)(object)BitConverter.ToInt32(bytes, start),
              { ElemType: ElemType.Int16, Lanes: 1 } => (T)(object)BitConverter.ToInt16(bytes, start),
              { ElemType: ElemType.Int8, Lanes: 1 } => (T)(object)bytes[start],
              { ElemType: ElemType.UInt64, Lanes: 1 } => (T)(object)BitConverter.ToUInt64(bytes, start),
              { ElemType: ElemType.UInt32, Lanes: 1 } => (T)(object)BitConverter.ToUInt32(bytes, start),
              { ElemType: ElemType.UInt16, Lanes: 1 } => (T)(object)BitConverter.ToUInt16(bytes, start),
              { ElemType: ElemType.UInt8, Lanes: 1 } => (T)(object)bytes[start],
              { ElemType: ElemType.Float64, Lanes: 1 } => (T)(object)BitConverter.ToDouble(bytes, start),
              { ElemType: ElemType.Float32, Lanes: 1 } => (T)(object)BitConverter.ToSingle(bytes, start),
              { ElemType: ElemType.BFloat16, Lanes: 1 } => (T)(object)(new BFloat16(bytes[start])),
              { ElemType: ElemType.Float16, Lanes: 1 } => (T)(object)(Half)(bytes[start]),
              { ElemType: ElemType.Bool, Lanes: 1 } => (T)(object)BitConverter.ToBoolean(bytes, start),
              { ElemType: ElemType.String, Lanes: 1 } => (T)(object)BitConverter.ToString(bytes, start),
              _ => throw new InvalidCastException($"Can't Convert the {srcType.ToString()}!")
          };

        public static T CastToScalar<T>(DataType srcType, byte[] bytes, int start = 0)
        where T : unmanaged
        => srcType switch
        {
            { ElemType: ElemType.Int64, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToInt64(bytes, start), typeof(T)),
            { ElemType: ElemType.Int32, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToInt32(bytes, start), typeof(T)),
            { ElemType: ElemType.Int16, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToInt16(bytes, start), typeof(T)),
            { ElemType: ElemType.Int8, Lanes: 1 } => (T)Convert.ChangeType((object)bytes[start], typeof(T)),
            { ElemType: ElemType.UInt64, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToUInt64(bytes, start), typeof(T)),
            { ElemType: ElemType.UInt32, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToUInt32(bytes, start), typeof(T)),
            { ElemType: ElemType.UInt16, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToUInt16(bytes, start), typeof(T)),
            { ElemType: ElemType.UInt8, Lanes: 1 } => (T)Convert.ChangeType((object)bytes[start], typeof(T)),
            { ElemType: ElemType.Float64, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToDouble(bytes, start), typeof(T)),
            { ElemType: ElemType.Float32, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToSingle(bytes, start), typeof(T)),
            { ElemType: ElemType.BFloat16, Lanes: 1 } => (T)Convert.ChangeType((object)(new BFloat16(bytes[start])), typeof(T)),
            { ElemType: ElemType.Float16, Lanes: 1 } => (T)Convert.ChangeType((object)(Half)(bytes[start]), typeof(T)),
            { ElemType: ElemType.Bool, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToBoolean(bytes, start), typeof(T)),
            _ => throw new InvalidCastException($"Can't Cast the {srcType.ToString()}!")
        };

        public static bool IsIntegral(DataType srcType, int Lanes = 1) =>
          srcType.ElemType switch
          {
              (ElemType.Bool or
               ElemType.Int64 or ElemType.Int32 or ElemType.Int16 or ElemType.Int8 or
               ElemType.UInt64 or ElemType.UInt32 or ElemType.UInt16 or ElemType.UInt8) => true,
              _ => false
          } && Lanes == srcType.Lanes;

        public static bool IsFloat(DataType srcType, int Lanes = 1) => srcType.ElemType switch
        {
            (ElemType.BFloat16 or ElemType.Float16 or ElemType.Float32 or ElemType.Float64) => true,
            _ => false
        } && Lanes == srcType.Lanes;
    }
}
