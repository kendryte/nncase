// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime;

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
/// PrimTypeCode helper.
/// </summary>
public static class PrimTypeCodes
{
    private static readonly Dictionary<DataType, PrimTypeCode> _primTypeCodes = new()
    {
        { DataTypes.UInt8, PrimTypeCode.UInt8 },
        { DataTypes.UInt16, PrimTypeCode.UInt16 },
        { DataTypes.UInt32, PrimTypeCode.UInt32 },
        { DataTypes.UInt64, PrimTypeCode.UInt64 },
        { DataTypes.Int8, PrimTypeCode.Int8 },
        { DataTypes.Int16, PrimTypeCode.Int16 },
        { DataTypes.Int32, PrimTypeCode.Int32 },
        { DataTypes.Int64, PrimTypeCode.Int64 },
        { DataTypes.BFloat16, PrimTypeCode.BFloat16 },
        { DataTypes.Float16, PrimTypeCode.Float16 },
        { DataTypes.Float32, PrimTypeCode.Float32 },
        { DataTypes.Float64, PrimTypeCode.Float64 },
    };

    private static readonly Dictionary<PrimTypeCode, DataType> _dataTypes = new()
    {
        { PrimTypeCode.UInt8, DataTypes.UInt8 },
        { PrimTypeCode.UInt16, DataTypes.UInt16 },
        { PrimTypeCode.UInt32, DataTypes.UInt32 },
        { PrimTypeCode.UInt64, DataTypes.UInt64 },
        { PrimTypeCode.Int8, DataTypes.Int8 },
        { PrimTypeCode.Int16, DataTypes.Int16 },
        { PrimTypeCode.Int32, DataTypes.Int32 },
        { PrimTypeCode.Int64, DataTypes.Int64 },
    };

    /// <summary>
    /// convert datatype to typecode.
    /// </summary>
    /// <param name="dataType">datatype.</param>
    /// <returns>primtype code.</returns>
    public static PrimTypeCode ToTypeCode(DataType dataType)
    {
        return _primTypeCodes[dataType];
    }

    /// <summary>
    /// convert primtype code to datatype.
    /// </summary>
    /// <param name="typeCode">prim type code.</param>
    /// <returns>datatype.</returns>
    public static DataType ToDataType(PrimTypeCode typeCode)
    {
        return _dataTypes[typeCode];
    }
}
