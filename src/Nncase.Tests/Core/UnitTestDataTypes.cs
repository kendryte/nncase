// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using Nncase;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestDataTypes
{
    [Fact]
    public void TestIsIntegral()
    {
        Assert.False(DataTypes.IsIntegral(DataTypes.Float16));
        Assert.False(DataTypes.IsIntegral(DataTypes.Float32));
        Assert.False(DataTypes.IsIntegral(DataTypes.Float64));
        Assert.False(DataTypes.IsIntegral(DataTypes.BFloat16));
        Assert.False(DataTypes.IsIntegral(DataTypes.Boolean));
        Assert.False(DataTypes.IsIntegral(DataTypes.Utf8Char));
        Assert.True(DataTypes.IsIntegral(DataTypes.Int8));
        Assert.True(DataTypes.IsIntegral(DataTypes.UInt8));
        Assert.True(DataTypes.IsIntegral(DataTypes.Int16));
        Assert.True(DataTypes.IsIntegral(DataTypes.UInt16));
        Assert.True(DataTypes.IsIntegral(DataTypes.Int32));
        Assert.True(DataTypes.IsIntegral(DataTypes.UInt32));
        Assert.True(DataTypes.IsIntegral(DataTypes.Int64));
        Assert.True(DataTypes.IsIntegral(DataTypes.UInt64));
    }

    [Fact]
    public void TestIsFloat()
    {
        Assert.False(DataTypes.IsFloat(DataTypes.Int8));
        Assert.False(DataTypes.IsFloat(DataTypes.UInt8));
        Assert.False(DataTypes.IsFloat(DataTypes.Int16));
        Assert.False(DataTypes.IsFloat(DataTypes.UInt16));
        Assert.False(DataTypes.IsFloat(DataTypes.Int32));
        Assert.False(DataTypes.IsFloat(DataTypes.UInt32));
        Assert.False(DataTypes.IsFloat(DataTypes.Int64));
        Assert.False(DataTypes.IsFloat(DataTypes.UInt64));
        Assert.False(DataTypes.IsFloat(DataTypes.Boolean));
        Assert.False(DataTypes.IsFloat(DataTypes.Utf8Char));
        Assert.True(DataTypes.IsFloat(DataTypes.Float16));
        Assert.True(DataTypes.IsFloat(DataTypes.Float32));
        Assert.True(DataTypes.IsFloat(DataTypes.Float64));
        Assert.True(DataTypes.IsFloat(DataTypes.BFloat16));
    }

    [Fact]
    public void TestIsPointer()
    {
        Assert.False(DataTypes.IsPointer(DataTypes.Float32));
        Assert.True(DataTypes.IsPointer(new PointerType(DataTypes.Float32)));
    }

    [Fact]
    public void TestGetDisplayName()
    {
        var a = new QuantParamType();
        Assert.Equal(a.ToString(), DataTypes.GetDisplayName(a));
        Assert.Equal("(f32*)", DataTypes.GetDisplayName(new PointerType(DataTypes.Float32)));
        Assert.Equal(DataTypes.Boolean.ShortName, DataTypes.GetDisplayName(DataTypes.Boolean));
        Assert.Equal(DataTypes.Utf8Char.ShortName, DataTypes.GetDisplayName(DataTypes.Utf8Char));
        Assert.Equal(DataTypes.Int8.ShortName, DataTypes.GetDisplayName(DataTypes.Int8));
        Assert.Equal(DataTypes.UInt8.ShortName, DataTypes.GetDisplayName(DataTypes.UInt8));
        Assert.Equal(DataTypes.Int16.ShortName, DataTypes.GetDisplayName(DataTypes.Int16));
        Assert.Equal(DataTypes.UInt16.ShortName, DataTypes.GetDisplayName(DataTypes.UInt16));
        Assert.Equal(DataTypes.Int32.ShortName, DataTypes.GetDisplayName(DataTypes.Int32));
        Assert.Equal(DataTypes.UInt32.ShortName, DataTypes.GetDisplayName(DataTypes.UInt32));
        Assert.Equal(DataTypes.Int64.ShortName, DataTypes.GetDisplayName(DataTypes.Int64));
        Assert.Equal(DataTypes.UInt64.ShortName, DataTypes.GetDisplayName(DataTypes.UInt64));
        Assert.Equal(DataTypes.Float16.ShortName, DataTypes.GetDisplayName(DataTypes.Float16));
        Assert.Equal(DataTypes.Float32.ShortName, DataTypes.GetDisplayName(DataTypes.Float32));
        Assert.Equal(DataTypes.Float64.ShortName, DataTypes.GetDisplayName(DataTypes.Float64));
        Assert.Equal(DataTypes.BFloat16.ShortName, DataTypes.GetDisplayName(DataTypes.BFloat16));
    }
}
