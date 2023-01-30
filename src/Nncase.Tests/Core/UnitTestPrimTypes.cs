// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using Nncase;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestPrimTypes
{
    public static unsafe IEnumerable<object[]> TestPrimTypesData =>
        new[]
        {
            new object[] { new BooleanType(), PrimTypeAttributes.None, "Boolean", "bool", typeof(bool), sizeof(bool), Runtime.TypeCode.Boolean },
            new object[] { new Utf8CharType(), PrimTypeAttributes.None, "Utf8Char", "u8char", typeof(Utf8Char), sizeof(Utf8Char), Runtime.TypeCode.Utf8Char },
            new object[] { new Int8Type(), PrimTypeAttributes.IsInteger, "Int8", "i8", typeof(sbyte), sizeof(sbyte), Runtime.TypeCode.Int8 },
            new object[] { new UInt8Type(), PrimTypeAttributes.IsInteger, "UInt8", "u8", typeof(byte), sizeof(byte), Runtime.TypeCode.UInt8 },
            new object[] { new Int16Type(), PrimTypeAttributes.IsInteger, "Int16", "i16", typeof(short), sizeof(short), Runtime.TypeCode.Int16 },
            new object[] { new UInt16Type(), PrimTypeAttributes.IsInteger, "UInt16", "u16", typeof(ushort), sizeof(ushort), Runtime.TypeCode.UInt16 },
            new object[] { new Int32Type(), PrimTypeAttributes.IsInteger, "Int32", "i32", typeof(int), sizeof(int), Runtime.TypeCode.Int32 },
            new object[] { new UInt32Type(), PrimTypeAttributes.IsInteger, "UInt32", "u32", typeof(uint), sizeof(uint), Runtime.TypeCode.UInt32 },
            new object[] { new Int64Type(), PrimTypeAttributes.IsInteger, "Int64", "i64", typeof(long), sizeof(long), Runtime.TypeCode.Int64 },
            new object[] { new UInt64Type(), PrimTypeAttributes.IsInteger, "UInt64", "u64", typeof(ulong), sizeof(ulong), Runtime.TypeCode.UInt64 },
            new object[] { new Float16Type(), PrimTypeAttributes.IsFloat, "Float16", "f16", typeof(Half), sizeof(Half), Runtime.TypeCode.Float16 },
            new object[] { new Float32Type(), PrimTypeAttributes.IsFloat, "Float32", "f32", typeof(float), sizeof(float), Runtime.TypeCode.Float32 },
            new object[] { new Float64Type(), PrimTypeAttributes.IsFloat, "Float64", "f64", typeof(double), sizeof(double), Runtime.TypeCode.Float64 },
            new object[] { new BFloat16Type(), PrimTypeAttributes.IsFloat, "BFloat16", "bf16", typeof(BFloat16), sizeof(BFloat16), Runtime.TypeCode.BFloat16 },
        };

    [Theory]
    [MemberData(nameof(TestPrimTypesData))]
    public void TestPrimTypes(PrimType a, PrimTypeAttributes attr, string fullName, string shortName, Type clrType, int sizeInBytes, Runtime.TypeCode typeCode)
    {
        Assert.Equal(attr, a.Attributes);
        Assert.Equal(fullName, a.FullName);
        Assert.Equal(shortName, a.ShortName);
        Assert.Equal(clrType, a.CLRType);
        Assert.Equal(sizeInBytes, a.SizeInBytes);
        Assert.Equal(typeCode, a.TypeCode);
    }
}
