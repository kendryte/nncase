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
            var generic = t.GetGenericTypeDefinition();
            if (generic == typeof(Pointer<>))
            {
                return new PointerType(FromType(t.GenericTypeArguments[0]));
            }
            else if (generic == typeof(Vector4<>))
            {
                return new VectorType(FromType(t.GenericTypeArguments[0]), 4);
            }
            else if (generic == typeof(Vector4x4<>))
            {
                return new VectorType(FromType(t.GenericTypeArguments[0]), 4, 4);
            }
            else if (generic == typeof(Vector8<>))
            {
                return new VectorType(FromType(t.GenericTypeArguments[0]), 8);
            }
            else if (generic == typeof(Vector16<>))
            {
                return new VectorType(FromType(t.GenericTypeArguments[0]), 16);
            }
            else if (generic == typeof(Vector16x16<>))
            {
                return new VectorType(FromType(t.GenericTypeArguments[0]), 16, 16);
            }
            else if (generic == typeof(Vector32<>))
            {
                return new VectorType(FromType(t.GenericTypeArguments[0]), 32);
            }
            else if (generic == typeof(Vector32x16<>))
            {
                return new VectorType(FromType(t.GenericTypeArguments[0]), 32, 16);
            }
            else if (generic == typeof(Vector32x32<>))
            {
                return new VectorType(FromType(t.GenericTypeArguments[0]), 32, 32);
            }
            else if (generic == typeof(Vector32x64<>))
            {
                return new VectorType(FromType(t.GenericTypeArguments[0]), 32, 64);
            }
            else if (generic == typeof(Vector64<>))
            {
                return new VectorType(FromType(t.GenericTypeArguments[0]), 64);
            }
            else if (generic == typeof(Vector128<>))
            {
                return new VectorType(FromType(t.GenericTypeArguments[0]), 128);
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

/// <summary>
/// Vector type.
/// </summary>
public sealed record VectorType(DataType ElemType, IR.IRArray<int> Lanes) : DataType
{
    public VectorType(DataType elemType, params int[] lanes)
        : this(elemType, new IR.IRArray<int>(lanes))
    {
    }

    public override Type CLRType => Lanes.ToArray() switch
    {
        [4] => typeof(Vector4<>).MakeGenericType(ElemType.CLRType),
        [4, 4] => typeof(Vector4x4<>).MakeGenericType(ElemType.CLRType),
        [8] => typeof(Vector8<>).MakeGenericType(ElemType.CLRType),
        [8, 8] => typeof(Vector8x8<>).MakeGenericType(ElemType.CLRType),
        [16] => typeof(Vector16<>).MakeGenericType(ElemType.CLRType),
        [16, 16] => typeof(Vector16x16<>).MakeGenericType(ElemType.CLRType),
        [32] => typeof(Vector32<>).MakeGenericType(ElemType.CLRType),
        [32, 16] => typeof(Vector32x16<>).MakeGenericType(ElemType.CLRType),
        [32, 32] => typeof(Vector32x32<>).MakeGenericType(ElemType.CLRType),
        [32, 64] => typeof(Vector32x64<>).MakeGenericType(ElemType.CLRType),
        [64] => typeof(Vector64<>).MakeGenericType(ElemType.CLRType),
        [128] => typeof(Vector128<>).MakeGenericType(ElemType.CLRType),
        _ => throw new NotSupportedException(),
    };

    public override int SizeInBytes => ElemType.SizeInBytes * (int)TensorUtilities.GetProduct(Lanes.ToArray());
}
