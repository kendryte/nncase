// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime.Interop;

/// <summary>
/// the runtime interop extensions method.
/// </summary>
public static class RTExtensions
{
    /// <summary>
    /// convert the rt datatype to primtype.
    /// </summary>
    public static PrimType ToPrimType(this RTDataType rt_dtype) => rt_dtype.TypeCode
    switch
    {
        TypeCode.Boolean => DataTypes.Boolean,
        TypeCode.Int8 => DataTypes.Int8,
        TypeCode.Int16 => DataTypes.Int16,
        TypeCode.Int32 => DataTypes.Int32,
        TypeCode.Int64 => DataTypes.Int64,
        TypeCode.UInt8 => DataTypes.UInt8,
        TypeCode.UInt16 => DataTypes.UInt16,
        TypeCode.UInt32 => DataTypes.UInt32,
        TypeCode.UInt64 => DataTypes.UInt64,
        TypeCode.Float16 => DataTypes.Float16,
        TypeCode.Float32 => DataTypes.Float32,
        TypeCode.Float64 => DataTypes.Float64,
        TypeCode.BFloat16 => DataTypes.BFloat16,
        _ => throw new ArgumentOutOfRangeException(nameof(rt_dtype)),
    };
}
