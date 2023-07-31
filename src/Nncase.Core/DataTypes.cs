// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// Data types.
/// </summary>
public static class DataTypes
{
    /// <summary>
    /// structcut for Int8.
    /// </summary>
    public static readonly PrimType Int8 = new Int8Type();

    /// <summary>
    /// structcut for Int16.
    /// </summary>
    public static readonly PrimType Int16 = new Int16Type();

    /// <summary>
    /// structcut for Int32.
    /// </summary>
    public static readonly PrimType Int32 = new Int32Type();

    /// <summary>
    /// structcut for Int64.
    /// </summary>
    public static readonly PrimType Int64 = new Int64Type();

    /// <summary>
    /// structcut for UInt8.
    /// </summary>
    public static readonly PrimType UInt8 = new UInt8Type();

    /// <summary>
    /// structcut for UInt16.
    /// </summary>
    public static readonly PrimType UInt16 = new UInt16Type();

    /// <summary>
    /// structcut for UInt32.
    /// </summary>
    public static readonly PrimType UInt32 = new UInt32Type();

    /// <summary>
    /// structcut for UInt64.
    /// </summary>
    public static readonly PrimType UInt64 = new UInt64Type();

    /// <summary>
    /// structcut for Float16.
    /// </summary>
    public static readonly PrimType Float16 = new Float16Type();

    /// <summary>
    /// structcut for Float32.
    /// </summary>
    public static readonly PrimType Float32 = new Float32Type();

    /// <summary>
    /// structcut for Float64.
    /// </summary>
    public static readonly PrimType Float64 = new Float64Type();

    /// <summary>
    /// structcut for BFloat16.
    /// </summary>
    public static readonly PrimType BFloat16 = new BFloat16Type();

    /// <summary>
    /// structcut for Boolean.
    /// </summary>
    public static readonly PrimType Boolean = new BooleanType();

    /// <summary>
    /// structcut for Utf8 Char.
    /// </summary>
    public static readonly PrimType Utf8Char = new Utf8CharType();

    /// <summary>
    /// check the datatype is lanes.
    /// </summary>
    /// <param name="srcType">Data type.</param>
    /// <returns>Checked result.</returns>
    public static bool IsIntegral(this DataType srcType) =>
      srcType is PrimType ptype && ptype.Attributes.HasFlag(PrimTypeAttributes.IsInteger);

    /// <summary>
    /// check the data type is float.
    /// </summary>
    /// <param name="srcType">Data type.</param>
    /// <returns>Checked result.</returns>
    public static bool IsFloat(this DataType srcType) =>
      srcType is PrimType ptype && ptype.Attributes.HasFlag(PrimTypeAttributes.IsFloat);

    /// <summary>
    /// check the data type is pointer.
    /// </summary>
    /// <param name="srcType">Data type.</param>
    /// <returns>Checked result.</returns>
    public static bool IsPointer(this DataType srcType) =>
      srcType is PointerType;

    /// <summary>
    /// display the datatype for il.
    /// </summary>
    /// <returns> datatype name.</returns>
    public static string GetDisplayName(this DataType dataType) => dataType switch
    {
        PointerType pointerType => $"({GetDisplayName(pointerType.ElemType)}{(pointerType.Shape.IsScalar ? string.Empty : pointerType.Shape.ToString())} *)",
        PrimType primType => primType.ShortName,
        ValueType => dataType.ToString(),
        _ => throw new ArgumentOutOfRangeException(dataType.GetType().Name),
    };

    /// <summary>
    /// display the datatype for csharp build.
    /// </summary>
    /// <param name="dataType">datatype.</param>
    /// <returns>datatype ctor.</returns>
    public static string GetCSharpName(this DataType dataType) => dataType switch
    {
        PrimType primType => $"DataTypes.{primType.FullName}",
        PointerType pointerType => $"new PointerType({pointerType.ElemType.GetCSharpName()}, IR.Shape.Scalar)",
        ValueType valueType => $"new {valueType.GetType().Name}()",
        _ => throw new ArgumentOutOfRangeException(dataType.GetType().Name),
    };

    /// <summary>
    /// display the datatype for builtin name. eg. Single => float.
    /// </summary>
    /// <param name="dataType">datatype.</param>
    /// <returns>builtin name.</returns>
    public static string GetBuiltInName(this DataType dataType) => dataType switch
    {
        PrimType primType => primType.TypeCode switch
        {
            Runtime.TypeCode.Boolean => "bool",
            Runtime.TypeCode.Utf8Char => "Utf8Char",
            Runtime.TypeCode.Int8 => "sbyte",
            Runtime.TypeCode.Int16 => "short",
            Runtime.TypeCode.Int32 => "int",
            Runtime.TypeCode.Int64 => "long",
            Runtime.TypeCode.UInt8 => "byte",
            Runtime.TypeCode.UInt16 => "ushort",
            Runtime.TypeCode.UInt32 => "uint",
            Runtime.TypeCode.UInt64 => "ulong",
            Runtime.TypeCode.Float16 => "Half",
            Runtime.TypeCode.Float32 => "float",
            Runtime.TypeCode.Float64 => "double",
            Runtime.TypeCode.BFloat16 => "BFloat16",
            _ => throw new ArgumentOutOfRangeException(primType.FullName),
        },
        PointerType pointerType => throw new NotSupportedException(nameof(PointerType)),
        ValueType valueType => $"{valueType.CLRType.Name}",
        _ => throw new ArgumentOutOfRangeException(dataType.GetType().Name),
    };

    public static DataType FromShortName(string shortName) => shortName switch
    {
        var x when x == DataTypes.UInt8.ShortName => DataTypes.UInt8,
        var x when x == DataTypes.Int8.ShortName => DataTypes.Int8,
        var x when x == DataTypes.Int16.ShortName => DataTypes.Int16,
        var x when x == DataTypes.Int32.ShortName => DataTypes.Int32,
        var x when x == DataTypes.Float16.ShortName => DataTypes.Float16,
        var x when x == DataTypes.Float32.ShortName => DataTypes.Float32,
        var x when x == DataTypes.BFloat16.ShortName => DataTypes.BFloat16,
        _ => throw new ArgumentException("Invalid data type."),
    };
}
