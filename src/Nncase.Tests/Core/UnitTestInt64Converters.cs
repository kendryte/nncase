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

public sealed class UnitTestInt64Converters
{
    public static unsafe IEnumerable<object[]> TestBoolData =>
        new[]
        {
            new object[] { new bool[] { true, false, true, false }, new long[] { 1, 0, 3, 0 }, CastMode.KDefault },
            new object[] { new bool[] { true, false, true, false }, new long[] { 1, 0, 3, 0 }, CastMode.CheckOverflow },
            new object[] { new bool[] { true, false, true, false }, new long[] { 1, 0, 3, 0 }, CastMode.Reinterpret },
        };

    [Theory]
    [MemberData(nameof(TestBoolData))]
    public void TestConvertToBool(bool[] expected, long[] src, CastMode mode)
    {
        var actual = new bool[src.Length];
        var c = new Int64Converters();
        c.ConvertTo(src, actual, mode);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TestConvertToBoolException()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var actual = new bool[a.Length - 1];
        var c = new Int64Converters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual, CastMode.KDefault));
    }

    [Fact]
    public void TestConvertToSbyte()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var expected = a.Select(x => (sbyte)x).ToArray();
        var actual = new sbyte[a.Length];
        var c = new Int64Converters();

        c.ConvertTo(a, actual, CastMode.KDefault);
        Assert.Equal(expected, actual);

        c.ConvertTo(a, actual, CastMode.CheckOverflow);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TestConvertToSbyteException()
    {
        var a = new long[] { 1, 2, 3, long.MaxValue };
        var actual1 = new sbyte[a.Length];
        var actual2 = new sbyte[a.Length - 1];
        var c = new Int64Converters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
        Assert.Throws<OverflowException>(() => c.ConvertTo(a, actual1, CastMode.CheckOverflow));
    }

    [Fact]
    public void TestConvertToByte()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var expected = a.Select(x => (byte)x).ToArray();
        var actual = new byte[a.Length];
        var c = new Int64Converters();

        c.ConvertTo(a, actual, CastMode.KDefault);
        Assert.Equal(expected, actual);

        c.ConvertTo(a, actual, CastMode.CheckOverflow);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TestConvertToByteException()
    {
        var a = new long[] { 1, 2, 3, long.MaxValue };
        var actual1 = new byte[a.Length];
        var actual2 = new byte[a.Length - 1];
        var c = new Int64Converters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
        Assert.Throws<OverflowException>(() => c.ConvertTo(a, actual1, CastMode.CheckOverflow));
    }

    [Fact]
    public void TestConvertToShort()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var expected = a.Select(x => (short)x).ToArray();
        var actual = new short[a.Length];
        var c = new Int64Converters();

        c.ConvertTo(a, actual, CastMode.KDefault);
        Assert.Equal(expected, actual);

        c.ConvertTo(a, actual, CastMode.CheckOverflow);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TestConvertToShortException()
    {
        var a = new long[] { 1, 2, 3, long.MaxValue };
        var actual1 = new short[a.Length];
        var actual2 = new short[a.Length - 1];
        var c = new Int64Converters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
        Assert.Throws<OverflowException>(() => c.ConvertTo(a, actual1, CastMode.CheckOverflow));
    }

    [Fact]
    public void TestConvertToUshort()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var expected = a.Select(x => (ushort)x).ToArray();
        var actual = new ushort[a.Length];
        var c = new Int64Converters();

        c.ConvertTo(a, actual, CastMode.KDefault);
        Assert.Equal(expected, actual);

        c.ConvertTo(a, actual, CastMode.CheckOverflow);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TestConvertToUshortException()
    {
        var a = new long[] { 1, 2, 3, long.MinValue };
        var actual1 = new ushort[a.Length];
        var actual2 = new ushort[a.Length - 1];
        var c = new Int64Converters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
        Assert.Throws<OverflowException>(() => c.ConvertTo(a, actual1, CastMode.CheckOverflow));
    }

    [Fact]
    public void TestConvertToInt()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var expected = a.Select(x => (int)x).ToArray();
        var actual = new int[a.Length];
        var c = new Int64Converters();

        c.ConvertTo(a, actual, CastMode.KDefault);
        Assert.Equal(expected, actual);

        c.ConvertTo(a, actual, CastMode.CheckOverflow);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TestConvertToIntException()
    {
        var a = new long[] { 1, 2, 3, long.MinValue };
        var actual1 = new int[a.Length];
        var actual2 = new int[a.Length - 1];
        var c = new Int64Converters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
        Assert.Throws<OverflowException>(() => c.ConvertTo(a, actual1, CastMode.CheckOverflow));
    }

    [Fact]
    public void TestConvertToUint()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var expected = a.Select(x => (uint)x).ToArray();
        var actual = new uint[a.Length];
        var c = new Int64Converters();

        c.ConvertTo(a, actual, CastMode.KDefault);
        Assert.Equal(expected, actual);

        c.ConvertTo(a, actual, CastMode.CheckOverflow);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TestConvertToUintException()
    {
        var a = new long[] { 1, 2, 3, long.MinValue };
        var actual1 = new uint[a.Length];
        var actual2 = new uint[a.Length - 1];
        var c = new Int64Converters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
        Assert.Throws<OverflowException>(() => c.ConvertTo(a, actual1, CastMode.CheckOverflow));
    }

    [Fact]
    public void TestConvertToLong()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var expected = a;
        var actual = new long[a.Length];
        var c = new Int64Converters();

        c.ConvertTo(a, actual, CastMode.KDefault);
        Assert.Equal(expected, actual);

        c.ConvertTo(a, actual, CastMode.CheckOverflow);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TestConvertToLongException()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var actual = new long[a.Length - 1];
        var c = new Int64Converters();

        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual, CastMode.KDefault));
    }

    [Fact]
    public void TestConvertToUlong()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var expected = a.Select(x => (ulong)x).ToArray();
        var actual = new ulong[a.Length];
        var c = new Int64Converters();

        c.ConvertTo(a, actual, CastMode.KDefault);
        Assert.Equal(expected, actual);

        c.ConvertTo(a, actual, CastMode.CheckOverflow);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TestConvertToUlongException()
    {
        var a = new long[] { 1, 2, 3, long.MinValue };
        var actual1 = new ulong[a.Length];
        var actual2 = new ulong[a.Length - 1];
        var c = new Int64Converters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
        Assert.Throws<OverflowException>(() => c.ConvertTo(a, actual1, CastMode.CheckOverflow));
    }

    [Fact]
    public void TestConvertToHalf()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var expected = a.Select(x => (Half)x).ToArray();
        var actual = new Half[a.Length];
        var c = new Int64Converters();
        c.ConvertTo(a, actual, CastMode.KDefault);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TestConvertToHalfException()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var actual1 = new Half[a.Length];
        var actual2 = new Half[a.Length - 1];
        var c = new Int64Converters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
    }

    [Fact]
    public void TestConvertToFloat()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var expected = a.Select(x => (float)x).ToArray();
        var actual = new float[a.Length];
        var c = new Int64Converters();
        c.ConvertTo(a, actual, CastMode.KDefault);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TestConvertToFloatException()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var actual1 = new float[a.Length];
        var actual2 = new float[a.Length - 1];
        var c = new Int64Converters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
    }

    [Fact]
    public void TestConvertToDouble()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var expected = a.Select(x => (double)x).ToArray();
        var actual = new double[a.Length];
        var c = new Int64Converters();
        c.ConvertTo(a, actual, CastMode.KDefault);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TestConvertToDoubleException()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var actual1 = new double[a.Length];
        var actual2 = new double[a.Length - 1];
        var c = new Int64Converters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
    }

    [Fact]
    public void TestConvertToBFloat16()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var expected = a.Select(x => (BFloat16)x).ToArray();
        var actual = new BFloat16[a.Length];
        var c = new Int64Converters();
        c.ConvertTo(a, actual, CastMode.KDefault);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TestConvertToBFloat16Exception()
    {
        var a = new long[] { 1, 2, 3, 4 };
        var actual1 = new BFloat16[a.Length];
        var actual2 = new BFloat16[a.Length - 1];
        var c = new Int64Converters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
    }
}
