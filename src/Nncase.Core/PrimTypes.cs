﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// Prim type of <see cref="bool"/>.
/// </summary>
public sealed record BooleanType : PrimType
{
    /// <inheritdoc/>
    public override PrimTypeAttributes Attributes => PrimTypeAttributes.None;

    /// <inheritdoc/>
    public override string FullName => "Boolean";

    /// <inheritdoc/>
    public override string ShortName => "bool";

    /// <inheritdoc/>
    public override Type CLRType => typeof(bool);

    /// <inheritdoc/>
    public override int SizeInBytes => sizeof(bool);

    /// <inheritdoc/>
    public override Runtime.TypeCode TypeCode => Runtime.TypeCode.Boolean;
}

/// <summary>
/// Prim type of utf8 char.
/// </summary>
public sealed record Utf8CharType : PrimType
{
    /// <inheritdoc/>
    public override PrimTypeAttributes Attributes => PrimTypeAttributes.None;

    /// <inheritdoc/>
    public override string FullName => "Utf8Char";

    /// <inheritdoc/>
    public override string ShortName => "u8char";

    /// <inheritdoc/>
    public override Type CLRType => typeof(Utf8Char);

    /// <inheritdoc/>
    public unsafe override int SizeInBytes => sizeof(Utf8Char);

    /// <inheritdoc/>
    public override Runtime.TypeCode TypeCode => Runtime.TypeCode.Utf8Char;
}

/// <summary>
/// Prim type of <see cref="sbyte"/>.
/// </summary>
public sealed record Int8Type : PrimType
{
    /// <inheritdoc/>
    public override PrimTypeAttributes Attributes => PrimTypeAttributes.IsInteger;

    /// <inheritdoc/>
    public override string FullName => "Int8";

    /// <inheritdoc/>
    public override string ShortName => "i8";

    /// <inheritdoc/>
    public override Type CLRType => typeof(sbyte);

    /// <inheritdoc/>
    public override int SizeInBytes => sizeof(sbyte);

    /// <inheritdoc/>
    public override Runtime.TypeCode TypeCode => Runtime.TypeCode.Int8;
}

/// <summary>
/// Prim type of <see cref="byte"/>.
/// </summary>
public sealed record UInt8Type : PrimType
{
    /// <inheritdoc/>
    public override PrimTypeAttributes Attributes => PrimTypeAttributes.IsInteger;

    /// <inheritdoc/>
    public override string FullName => "UInt8";

    /// <inheritdoc/>
    public override string ShortName => "u8";

    /// <inheritdoc/>
    public override Type CLRType => typeof(byte);

    /// <inheritdoc/>
    public override int SizeInBytes => sizeof(byte);

    /// <inheritdoc/>
    public override Runtime.TypeCode TypeCode => Runtime.TypeCode.UInt8;
}

/// <summary>
/// Prim type of <see cref="short"/>.
/// </summary>
public sealed record Int16Type : PrimType
{
    /// <inheritdoc/>
    public override PrimTypeAttributes Attributes => PrimTypeAttributes.IsInteger;

    /// <inheritdoc/>
    public override string FullName => "Int16";

    /// <inheritdoc/>
    public override string ShortName => "i16";

    /// <inheritdoc/>
    public override Type CLRType => typeof(short);

    /// <inheritdoc/>
    public override int SizeInBytes => sizeof(short);

    /// <inheritdoc/>
    public override Runtime.TypeCode TypeCode => Runtime.TypeCode.Int16;
}

/// <summary>
/// Prim type of <see cref="ushort"/>.
/// </summary>
public sealed record UInt16Type : PrimType
{
    /// <inheritdoc/>
    public override PrimTypeAttributes Attributes => PrimTypeAttributes.IsInteger;

    /// <inheritdoc/>
    public override string FullName => "UInt16";

    /// <inheritdoc/>
    public override string ShortName => "u16";

    /// <inheritdoc/>
    public override Type CLRType => typeof(ushort);

    /// <inheritdoc/>
    public override int SizeInBytes => sizeof(ushort);

    /// <inheritdoc/>
    public override Runtime.TypeCode TypeCode => Runtime.TypeCode.UInt16;
}

/// <summary>
/// Prim type of <see cref="int"/>.
/// </summary>
public sealed record Int32Type : PrimType
{
    /// <inheritdoc/>
    public override PrimTypeAttributes Attributes => PrimTypeAttributes.IsInteger;

    /// <inheritdoc/>
    public override string FullName => "Int32";

    /// <inheritdoc/>
    public override string ShortName => "i32";

    /// <inheritdoc/>
    public override Type CLRType => typeof(int);

    /// <inheritdoc/>
    public override int SizeInBytes => sizeof(int);

    /// <inheritdoc/>
    public override Runtime.TypeCode TypeCode => Runtime.TypeCode.Int32;
}

/// <summary>
/// Prim type of <see cref="uint"/>.
/// </summary>
public sealed record UInt32Type : PrimType
{
    /// <inheritdoc/>
    public override PrimTypeAttributes Attributes => PrimTypeAttributes.IsInteger;

    /// <inheritdoc/>
    public override string FullName => "UInt32";

    /// <inheritdoc/>
    public override string ShortName => "u32";

    /// <inheritdoc/>
    public override Type CLRType => typeof(uint);

    /// <inheritdoc/>
    public override int SizeInBytes => sizeof(uint);

    /// <inheritdoc/>
    public override Runtime.TypeCode TypeCode => Runtime.TypeCode.UInt32;
}

/// <summary>
/// Prim type of <see cref="long"/>.
/// </summary>
public sealed record Int64Type : PrimType
{
    /// <inheritdoc/>
    public override PrimTypeAttributes Attributes => PrimTypeAttributes.IsInteger;

    /// <inheritdoc/>
    public override string FullName => "Int64";

    /// <inheritdoc/>
    public override string ShortName => "i64";

    /// <inheritdoc/>
    public override Type CLRType => typeof(long);

    /// <inheritdoc/>
    public override int SizeInBytes => sizeof(long);

    /// <inheritdoc/>
    public override Runtime.TypeCode TypeCode => Runtime.TypeCode.Int64;
}

/// <summary>
/// Prim type of <see cref="ulong"/>.
/// </summary>
public sealed record UInt64Type : PrimType
{
    /// <inheritdoc/>
    public override PrimTypeAttributes Attributes => PrimTypeAttributes.IsInteger;

    /// <inheritdoc/>
    public override string FullName => "UInt64";

    /// <inheritdoc/>
    public override string ShortName => "u64";

    /// <inheritdoc/>
    public override Type CLRType => typeof(ulong);

    /// <inheritdoc/>
    public override int SizeInBytes => sizeof(ulong);

    /// <inheritdoc/>
    public override Runtime.TypeCode TypeCode => Runtime.TypeCode.UInt64;
}

/// <summary>
/// Prim type of <see cref="byte"/>.
/// </summary>
public sealed record Float8E4M3Type : PrimType
{
    /// <inheritdoc/>
    public override PrimTypeAttributes Attributes => PrimTypeAttributes.IsFloat;

    /// <inheritdoc/>
    public override string FullName => "Float8E4M3";

    /// <inheritdoc/>
    public override string ShortName => "F8_E4M3";

    /// <inheritdoc/>
    public override Type CLRType => typeof(Float8E4M3);

    /// <inheritdoc/>
    public unsafe override int SizeInBytes => sizeof(Float8E4M3);

    /// <inheritdoc/>
    public override Runtime.TypeCode TypeCode => Runtime.TypeCode.Float8E4M3;
}

/// <summary>
/// Prim type of <see cref="byte"/>.
/// </summary>
public sealed record Float8E5M2Type : PrimType
{
    /// <inheritdoc/>
    public override PrimTypeAttributes Attributes => PrimTypeAttributes.IsFloat;

    /// <inheritdoc/>
    public override string FullName => "Float8E5M2";

    /// <inheritdoc/>
    public override string ShortName => "F8_E5M2";

    /// <inheritdoc/>
    public override Type CLRType => typeof(Float8E5M2);

    /// <inheritdoc/>
    public unsafe override int SizeInBytes => sizeof(Float8E5M2);

    /// <inheritdoc/>
    public override Runtime.TypeCode TypeCode => Runtime.TypeCode.Float8E5M2;
}

/// <summary>
/// Prim type of <see cref="Half"/>.
/// </summary>
public sealed record Float16Type : PrimType
{
    /// <inheritdoc/>
    public override PrimTypeAttributes Attributes => PrimTypeAttributes.IsFloat;

    /// <inheritdoc/>
    public override string FullName => "Float16";

    /// <inheritdoc/>
    public override string ShortName => "f16";

    /// <inheritdoc/>
    public override Type CLRType => typeof(Half);

    /// <inheritdoc/>
    public unsafe override int SizeInBytes => sizeof(Half);

    /// <inheritdoc/>
    public override Runtime.TypeCode TypeCode => Runtime.TypeCode.Float16;
}

/// <summary>
/// Prim type of <see cref="float"/>.
/// </summary>
public sealed record Float32Type : PrimType
{
    /// <inheritdoc/>
    public override PrimTypeAttributes Attributes => PrimTypeAttributes.IsFloat;

    /// <inheritdoc/>
    public override string FullName => "Float32";

    /// <inheritdoc/>
    public override string ShortName => "f32";

    /// <inheritdoc/>
    public override Type CLRType => typeof(float);

    /// <inheritdoc/>
    public unsafe override int SizeInBytes => sizeof(float);

    /// <inheritdoc/>
    public override Runtime.TypeCode TypeCode => Runtime.TypeCode.Float32;
}

/// <summary>
/// Prim type of <see cref="double"/>.
/// </summary>
public sealed record Float64Type : PrimType
{
    /// <inheritdoc/>
    public override PrimTypeAttributes Attributes => PrimTypeAttributes.IsFloat;

    /// <inheritdoc/>
    public override string FullName => "Float64";

    /// <inheritdoc/>
    public override string ShortName => "f64";

    /// <inheritdoc/>
    public override Type CLRType => typeof(double);

    /// <inheritdoc/>
    public unsafe override int SizeInBytes => sizeof(double);

    /// <inheritdoc/>
    public override Runtime.TypeCode TypeCode => Runtime.TypeCode.Float64;
}

/// <summary>
/// Prim type of <see cref="BFloat16"/>.
/// </summary>
public sealed record BFloat16Type : PrimType
{
    /// <inheritdoc/>
    public override PrimTypeAttributes Attributes => PrimTypeAttributes.IsFloat;

    /// <inheritdoc/>
    public override string FullName => "BFloat16";

    /// <inheritdoc/>
    public override string ShortName => "bf16";

    /// <inheritdoc/>
    public override Type CLRType => typeof(BFloat16);

    /// <inheritdoc/>
    public unsafe override int SizeInBytes => sizeof(BFloat16);

    /// <inheritdoc/>
    public override Runtime.TypeCode TypeCode => Runtime.TypeCode.BFloat16;
}
