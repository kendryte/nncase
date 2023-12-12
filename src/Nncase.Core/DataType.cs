// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

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
    public override Type CLRType { get; } = typeof(Pointer<>).MakeGenericType(ElemType.CLRType);

    /// <inheritdoc/>
    public override int SizeInBytes => sizeof(ulong);
}

/// <summary>
/// the abstract datatype record.
/// </summary>
public abstract record DataType
{
    internal DataType()
    {
    }

    /// <summary>
    /// Gets CLR type.
    /// </summary>
    public abstract Type CLRType { get; }

    /// <summary>
    /// Gets size in bytes.
    /// </summary>
    public abstract int SizeInBytes { get; }

    /// <summary>
    /// Get data type from CLR type.
    /// </summary>
    /// <param name="t">CLR type.</param>
    /// <returns>Data type.</returns>
    public static DataType FromType(Type t)
    {
        if (t.IsGenericType)
        {
            if (t.GetGenericTypeDefinition() == typeof(Pointer<>))
            {
                return new PointerType(FromType(t.GenericTypeArguments[0]));
            }

            throw new ArgumentException("Unsupported CLR type.");
        }

        return CompilerServices.DataTypeService.GetDataTypeFromType(t);
    }

    /// <summary>
    /// Get data type from type code.
    /// </summary>
    /// <param name="typeCode">Type code.</param>
    /// <returns>Data type.</returns>
    public static PrimType FromTypeCode(Runtime.TypeCode typeCode)
    {
        return CompilerServices.DataTypeService.GetPrimTypeFromTypeCode(typeCode);
    }

    /// <summary>
    /// Get data type from CLR type.
    /// </summary>
    /// <typeparam name="T">CLR type.</typeparam>
    /// <returns>Data type.</returns>
    public static DataType FromType<T>()
        where T : unmanaged, IEquatable<T>
        => FromType(typeof(T));
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
    /// Gets typecode.
    /// </summary>
    public abstract Runtime.TypeCode TypeCode { get; }

    /// <inheritdoc/>
    public sealed override string ToString()
    {
        return ShortName;
    }
}

/// <summary>
/// Value type.
/// </summary>
public abstract record ValueType : DataType
{
    /// <summary>
    /// Gets uuid.
    /// </summary>
    public abstract Guid Uuid { get; }
}
