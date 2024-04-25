// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime;

/// <summary>
/// Type code.
/// </summary>
public enum TypeCode : byte
{
    /// <summary>
    /// <see cref="bool"/>.
    /// </summary>
    Boolean,

    /// <summary>
    /// <see cref="Nncase.Utf8Char"/>.
    /// </summary>
    Utf8Char,

    /// <summary>
    /// <see cref="sbyte"/>.
    /// </summary>
    Int8,

    /// <summary>
    /// <see cref="short"/>.
    /// </summary>
    Int16,

    /// <summary>
    /// <see cref="int"/>.
    /// </summary>
    Int32,

    /// <summary>
    /// <see cref="long"/>.
    /// </summary>
    Int64,

    /// <summary>
    /// <see cref="byte"/>.
    /// </summary>
    UInt8,

    /// <summary>
    /// <see cref="ushort"/>.
    /// </summary>
    UInt16,

    /// <summary>
    /// <see cref="uint"/>.
    /// </summary>
    UInt32,

    /// <summary>
    /// <see cref="ulong"/>.
    /// </summary>
    UInt64,

    /// <summary>
    /// <see cref="Half"/>.
    /// </summary>
    Float16,

    /// <summary>
    /// <see cref="float"/>.
    /// </summary>
    Float32,

    /// <summary>
    /// <see cref="double"/>.
    /// </summary>
    Float64,

    /// <summary>
    /// <see cref="BFloat16"/>.
    /// </summary>
    BFloat16,

    /// <summary>
    /// <see cref="Pointer{T}"/>.
    /// </summary>
    Pointer = 0xF0,

    /// <summary>
    /// <see cref="Nncase.ValueType"/>.
    /// </summary>
    ValueType = 0xF1,

    /// <summary>
    /// <see cref="Nncase.VectorType"/>.
    /// </summary>
    VectorType = 0xF2,
}
