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
}
