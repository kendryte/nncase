// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase;
using Nncase.Converters;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestPointerConverters
{
    [Fact]
    public unsafe void TestConvert()
    {
        var f1 = 1F;
        var f2 = 2F;
        var expected = new ulong[] { (ulong)&f1, (ulong)&f2 };
        var p1 = new Pointer<float>((ulong)&f1);
        var p2 = new Pointer<float>((ulong)&f2);
        var a = new Pointer<float>[] { p1, p2 };
        var actual = new ulong[a.Length];
        var c = new PointerConverters();
        c.ConvertTo<float>(a, actual, CastMode.KDefault);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public unsafe void TestConvertException()
    {
        var f1 = 1F;
        var f2 = 2F;
        var p1 = new Pointer<float>((ulong)&f1);
        var p2 = new Pointer<float>((ulong)&f2);
        var a = new Pointer<float>[] { p1, p2 };
        var actual1 = new ulong[a.Length];
        var actual2 = new ulong[a.Length - 1];
        var c = new PointerConverters();
        Assert.Throws<InvalidCastException>(() => c.ConvertTo<float>(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo<float>(a, actual2, CastMode.KDefault));
    }
}
