// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using Nncase;
using Xunit;
using TypeCode = Nncase.Runtime.TypeCode;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestDataType
{
    [Fact]
    public void TestPointerType()
    {
        var pType = new PointerType(DataTypes.Float32);
        Assert.Equal(8, pType.SizeInBytes);

        var t = DataType.FromType(typeof(Pointer<int>));
        Assert.Equal(8, t.SizeInBytes);
    }

    [Fact]
    public void TestFromTypeCode()
    {
        var expect = CompilerServices.DataTypeService.GetPrimTypeFromTypeCode(TypeCode.Boolean);
        Assert.Equal(expect, DataType.FromTypeCode(TypeCode.Boolean));

        Assert.Throws<NullReferenceException>(() => CompilerServices.DataTypeService.GetPrimTypeFromType(Type.GetType(string.Empty)!));
        Assert.Throws<NullReferenceException>(() => CompilerServices.DataTypeService.GetValueTypeFromType(Type.GetType(string.Empty)!));
    }
}
