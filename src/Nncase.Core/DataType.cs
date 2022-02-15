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

namespace Nncase;

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
    /// <inheritdoc/>
    public override string ToString()
    {
        return $"{ElemType}*";
    }
}

/// <summary>
/// the abstract datatype record.
/// </summary>
public abstract record DataType()
{
    /// <summary>
    /// Get data type from CLR type.
    /// </summary>
    /// <param name="t">CLR type.</param>
    /// <returns>Data type.</returns>
    public static PrimType FromType(Type t)
    {
        return CompilerServices.DataTypeService.GetPrimTypeFromType(t);
    }

    /// <summary>
    /// Get data type from CLR type.
    /// </summary>
    /// <typeparam name="T">CLR type.</typeparam>
    /// <returns>Data type.</returns>
    public static PrimType FromType<T>()
        where T : unmanaged, IEquatable<T>
        => FromType(typeof(T));
}

/// <summary>
/// Attributes of <see cref="PrimType"/>.
/// </summary>
[Flags]
public enum PrimTypeAttributes
{
    /// <summary>
    /// None.
    /// </summary>
    None = 0,

    /// <summary>
    /// Is integer.
    /// </summary>
    IsInteger = 1,

    /// <summary>
    /// Is floating point.
    /// </summary>
    IsFloat = 2,
}

/// <summary>
/// Prim type.
/// </summary>
public abstract record PrimType : DataType
{
    /// <summary>
    /// Gets attributes.
    /// </summary>
    public abstract PrimTypeAttributes Attributes { get; }

    /// <summary>
    /// Gets full name.
    /// </summary>
    public abstract string FullName { get; }

    /// <summary>
    /// Gets short name.
    /// </summary>
    public abstract string ShortName { get; }

    /// <summary>
    /// Gets CLR type.
    /// </summary>
    public abstract Type CLRType { get; }

    /// <summary>
    /// Gets size in bytes.
    /// </summary>
    public abstract int SizeInBytes { get; }

    /// <inheritdoc/>
    public override string ToString()
    {
        return ShortName;
    }
}
