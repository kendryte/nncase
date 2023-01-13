// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using Nncase;
using Xunit;

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
}
