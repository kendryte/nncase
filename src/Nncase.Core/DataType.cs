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
    public enum PrimTypeCode : byte
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
    /// The storge data Type, for simd/npu/gpu. 
    /// <example>
    /// float32*4
    /// int8*2
    /// </example>
    /// </summary>
    /// <param name="ElemType"> the type the pointer points to. </param>
    public sealed record PointerType(DataType ElemType) : DataType
    {
    }

    /// <summary>
    /// the abstract datatype record
    /// </summary>
    public abstract record DataType()
    {
        /// <summary>
        /// structcut for Int8
        /// </summary>
        public static PrimType Int8 => PrimTypeCode.Int8;
        /// <summary>
        /// structcut for Int16
        /// </summary>
        public static PrimType Int16 => PrimTypeCode.Int16;

        /// <summary>
        /// structcut for Int32
        /// </summary>
        public static PrimType Int32 => PrimTypeCode.Int32;

        /// <summary>
        /// structcut for Int64
        /// </summary>
        public static PrimType Int64 => PrimTypeCode.Int64;

        /// <summary>
        /// structcut for UInt8
        /// </summary>
        public static PrimType UInt8 => PrimTypeCode.UInt8;

        /// <summary>
        /// structcut for UInt16
        /// </summary>
        public static PrimType UInt16 => PrimTypeCode.UInt16;

        /// <summary>
        /// structcut for UInt32
        /// </summary>
        public static PrimType UInt32 => PrimTypeCode.UInt32;

        /// <summary>
        /// structcut for UInt64
        /// </summary>
        public static PrimType UInt64 => PrimTypeCode.UInt64;

        /// <summary>
        /// structcut for Float16
        /// </summary>
        public static PrimType Float16 => PrimTypeCode.Float16;

        /// <summary>
        /// structcut for Float32
        /// </summary>
        public static PrimType Float32 => PrimTypeCode.Float32;

        /// <summary>
        /// structcut for Float64
        /// </summary>
        public static PrimType Float64 => PrimTypeCode.Float64;

        /// <summary>
        /// structcut for BFloat16
        /// </summary>
        public static PrimType BFloat16 => PrimTypeCode.BFloat16;

        /// <summary>
        /// structcut for Bool
        /// </summary>
        public static PrimType Bool => PrimTypeCode.Bool;

        /// <summary>
        /// structcut for String
        /// </summary>
        public static PrimType String => PrimTypeCode.String;

        /// <summary>
        /// structcut for Invalid
        /// </summary>
        public static PrimType Invalid => PrimTypeCode.Invalid;

        /// <summary>
        /// check current compatible with other datatype
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public Compatible CompatibleWith(DataType other) => new Compatible(
          (this is PrimType ptype) && (other is PrimType otype) && (ptype.TypeCode == otype.TypeCode && ptype.Lanes % otype.Lanes == 0),
          $"this {this} != other {other}");
    }

    /// <summary>
    /// the datatype 
    /// </summary>
    /// <param name="TypeCode"></param>
    /// <param name="Lanes"></param>
    public record PrimType(PrimTypeCode TypeCode, int Lanes = 1) : DataType
    {
        /// <summary>
        /// from the PrimTypeCode
        /// </summary>
        /// <param name="TypeCode"></param>
        public static implicit operator PrimType(PrimTypeCode TypeCode) => new PrimType(TypeCode, 1);

        /// <summary>
        /// convert datatype to string
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return ("PrimType." + TypeCode) + (Lanes == 1 ? string.Empty : Lanes.ToString());
        }
    }

    /// <summary>
    /// the Compatible information.
    /// </summary>
    /// <param name="IsCompatible"></param>
    /// <param name="Reason"></param>
    public sealed record Compatible(bool IsCompatible, string Reason)
    {
        /// <summary>
        /// convert as bool
        /// </summary>
        /// <param name="compatible"></param>
        public static implicit operator bool(Compatible compatible) => compatible.IsCompatible;
    }

    /// <summary>
    /// Data type helper.
    /// </summary>
    public static class DataTypes
    {
        private static readonly Dictionary<RuntimeTypeHandle, PrimType> _typeToDataTypes = new()
        {
            { typeof(bool).TypeHandle, PrimType.Bool },
            { typeof(sbyte).TypeHandle, PrimType.Int8 },
            { typeof(byte).TypeHandle, PrimType.UInt8 },
            { typeof(int).TypeHandle, PrimType.Int32 },
            { typeof(uint).TypeHandle, PrimType.UInt32 },
            { typeof(long).TypeHandle, PrimType.Int64 },
            { typeof(ulong).TypeHandle, PrimType.UInt64 },
            { typeof(float).TypeHandle, PrimType.Float32 },
            { typeof(double).TypeHandle, PrimType.Float64 },
            { typeof(char).TypeHandle, PrimType.String },
            { typeof(BFloat16).TypeHandle, PrimType.Float16 }
        };

        private static readonly Dictionary<PrimType, Type> _dataTypesToType = new()
        {
            { PrimType.Bool, typeof(bool) },
            { PrimType.Int8, typeof(sbyte) },
            { PrimType.UInt8, typeof(byte) },
            { PrimType.Int32, typeof(int) },
            { PrimType.UInt32, typeof(uint) },
            { PrimType.Int64, typeof(long) },
            { PrimType.UInt64, typeof(ulong) },
            { PrimType.Float32, typeof(float) },
            { PrimType.Float64, typeof(double) },
        };

        private static readonly Dictionary<PrimTypeCode, int> _ElemTypeToLengths = new()
        {
            { PrimTypeCode.Bool, 1 },
            { PrimTypeCode.UInt8, 1 },
            { PrimTypeCode.UInt16, 2 },
            { PrimTypeCode.UInt32, 4 },
            { PrimTypeCode.UInt64, 8 },
            { PrimTypeCode.Int8, 1 },
            { PrimTypeCode.Int16, 2 },
            { PrimTypeCode.Int32, 4 },
            { PrimTypeCode.Int64, 8 },
            { PrimTypeCode.Float16, 2 },
            { PrimTypeCode.BFloat16, 2 },
            { PrimTypeCode.Float32, 4 },
            { PrimTypeCode.Float64, 8 }
        };

        /// <summary>
        /// Get data type from CLR type.
        /// </summary>
        /// <param name="t">CLR type.</param>
        /// <returns>Data type.</returns>
        public static PrimType FromType(Type t)
        {
            if (_typeToDataTypes.TryGetValue(t.TypeHandle, out var dataType))
            {
                return dataType;
            }

            throw new ArgumentOutOfRangeException("Unsupported CLR type: " + t.FullName);
        }

        /// <summary>
        /// Get data type convert to CLR type.
        /// </summary>
        /// <param name="t"> Nncase datatype</param>
        /// <returns>CLR type instance.</returns>
        public static Type ToType(DataType t)
        {
            if (t is PrimType ptype && _dataTypesToType.TryGetValue(ptype, out var type)) return type;
            if (t is PointerType { ElemType: PrimType etype } potype) return ToType(etype).MakeArrayType();
            throw new ArgumentOutOfRangeException("Unsupported DataType type: " + t);
        }

        /// <summary>
        /// Get data type from CLR type.
        /// </summary>
        /// <typeparam name="T">CLR type.</typeparam>
        /// <returns>Data type.</returns>
        public static PrimType FromType<T>()
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
        public static int GetLength(DataType dataType) => dataType switch
        {
            PrimType ptype => _ElemTypeToLengths[ptype.TypeCode] * ptype.Lanes,
            PointerType potype => _ElemTypeToLengths[PrimTypeCode.UInt64],
            _ => throw new NotSupportedException(dataType.GetType().Name),
        };

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
            if (dataType is PrimType ptype)
            {
                var name = typeof(PrimTypeCode).GetField(Enum.GetName(ptype.TypeCode)!)!.GetCustomAttribute<DisplayAttribute>()?.GetName() ?? ptype.ToString()!;
                if (ptype.Lanes > 1)
                {
                    name += $"x{ptype.Lanes}";
                }
                return name;
            }
            else if (dataType is PointerType potype)
            {
                return "*" + potype.ElemType.ToString();
            }
            throw new NotImplementedException($"DataType {dataType.GetType().Name}");
        }

        /// <summary>
        /// convert a bytes to T scalar.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="srcType"></param>
        /// <param name="bytes"></param>
        /// <param name="start"></param>
        /// <returns></returns>
        /// <exception cref="InvalidCastException"></exception>
        public static T ToScalar<T>(DataType srcType, byte[] bytes, int start = 0)
          where T : unmanaged
          => (PrimType)srcType switch
          {
              { TypeCode: PrimTypeCode.Int64, Lanes: 1 } => (T)(object)BitConverter.ToInt64(bytes, start),
              { TypeCode: PrimTypeCode.Int32, Lanes: 1 } => (T)(object)BitConverter.ToInt32(bytes, start),
              { TypeCode: PrimTypeCode.Int16, Lanes: 1 } => (T)(object)BitConverter.ToInt16(bytes, start),
              { TypeCode: PrimTypeCode.Int8, Lanes: 1 } => (T)(object)bytes[start],
              { TypeCode: PrimTypeCode.UInt64, Lanes: 1 } => (T)(object)BitConverter.ToUInt64(bytes, start),
              { TypeCode: PrimTypeCode.UInt32, Lanes: 1 } => (T)(object)BitConverter.ToUInt32(bytes, start),
              { TypeCode: PrimTypeCode.UInt16, Lanes: 1 } => (T)(object)BitConverter.ToUInt16(bytes, start),
              { TypeCode: PrimTypeCode.UInt8, Lanes: 1 } => (T)(object)bytes[start],
              { TypeCode: PrimTypeCode.Float64, Lanes: 1 } => (T)(object)BitConverter.ToDouble(bytes, start),
              { TypeCode: PrimTypeCode.Float32, Lanes: 1 } => (T)(object)BitConverter.ToSingle(bytes, start),
              { TypeCode: PrimTypeCode.BFloat16, Lanes: 1 } => (T)(object)(new BFloat16(bytes[start])),
              { TypeCode: PrimTypeCode.Float16, Lanes: 1 } => (T)(object)(Half)(bytes[start]),
              { TypeCode: PrimTypeCode.Bool, Lanes: 1 } => (T)(object)BitConverter.ToBoolean(bytes, start),
              { TypeCode: PrimTypeCode.String, Lanes: 1 } => (T)(object)BitConverter.ToString(bytes, start),
              _ => throw new InvalidCastException($"Can't Convert the {srcType.ToString()}!")
          };

        /// <summary>
        /// convert the datatype to T scalar.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="srcType"></param>
        /// <param name="bytes"></param>
        /// <param name="start"></param>
        /// <returns></returns>
        /// <exception cref="InvalidCastException"></exception>
        public static T CastToScalar<T>(DataType srcType, byte[] bytes, int start = 0)
        where T : unmanaged
        => (PrimType)srcType switch
        {
            { TypeCode: PrimTypeCode.Int64, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToInt64(bytes, start), typeof(T)),
            { TypeCode: PrimTypeCode.Int32, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToInt32(bytes, start), typeof(T)),
            { TypeCode: PrimTypeCode.Int16, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToInt16(bytes, start), typeof(T)),
            { TypeCode: PrimTypeCode.Int8, Lanes: 1 } => (T)Convert.ChangeType((object)bytes[start], typeof(T)),
            { TypeCode: PrimTypeCode.UInt64, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToUInt64(bytes, start), typeof(T)),
            { TypeCode: PrimTypeCode.UInt32, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToUInt32(bytes, start), typeof(T)),
            { TypeCode: PrimTypeCode.UInt16, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToUInt16(bytes, start), typeof(T)),
            { TypeCode: PrimTypeCode.UInt8, Lanes: 1 } => (T)Convert.ChangeType((object)bytes[start], typeof(T)),
            { TypeCode: PrimTypeCode.Float64, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToDouble(bytes, start), typeof(T)),
            { TypeCode: PrimTypeCode.Float32, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToSingle(bytes, start), typeof(T)),
            { TypeCode: PrimTypeCode.BFloat16, Lanes: 1 } => (T)Convert.ChangeType((object)(new BFloat16(bytes[start])), typeof(T)),
            { TypeCode: PrimTypeCode.Float16, Lanes: 1 } => (T)Convert.ChangeType((object)(Half)(bytes[start]), typeof(T)),
            { TypeCode: PrimTypeCode.Bool, Lanes: 1 } => (T)Convert.ChangeType((object)BitConverter.ToBoolean(bytes, start), typeof(T)),
            _ => throw new InvalidCastException($"Can't Cast the {srcType.ToString()}!")
        };

        /// <summary>
        /// specify half cast.
        /// </summary>
        /// <param name="bytes"></param>
        /// <param name="start"></param>
        /// <returns></returns>
        public static Half CastToScalar(byte[] bytes, int start = 0)
        {
            return BitConverter.ToHalf(bytes, start);
        }

        /// <summary>
        /// check the datatype is lanes
        /// </summary>
        /// <param name="srcType"></param>
        /// <param name="Lanes"></param>
        /// <returns></returns>
        public static bool IsIntegral(DataType srcType, int Lanes = 1) =>
          srcType is PrimType ptype && ptype.TypeCode switch
          {
              (PrimTypeCode.Bool or
               PrimTypeCode.Int64 or PrimTypeCode.Int32 or PrimTypeCode.Int16 or PrimTypeCode.Int8 or
               PrimTypeCode.UInt64 or PrimTypeCode.UInt32 or PrimTypeCode.UInt16 or PrimTypeCode.UInt8) => true,
              _ => false
          } && Lanes == ptype.Lanes;

        /// <summary>
        /// check the data type is float
        /// </summary>
        /// <param name="srcType"></param>
        /// <param name="Lanes"></param>
        /// <returns></returns>
        public static bool IsFloat(DataType srcType, int Lanes = 1) =>
          srcType is PrimType ptype && ptype.TypeCode switch
          {
              (PrimTypeCode.BFloat16 or PrimTypeCode.Float16 or PrimTypeCode.Float32 or PrimTypeCode.Float64) => true,
              _ => false
          } && Lanes == ptype.Lanes;
    }
}
