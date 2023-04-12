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

public sealed class UnitTestBooleanConverters
{
    [Fact]
    public void TestConvertToBool()
    {
        var a = new bool[] { true, false, false, true };
        var b = new bool[a.Length];
        Assert.NotEqual(a, b);

        var c = new BooleanConverters();
        c.ConvertTo(a, b, CastMode.KDefault);
        Assert.Equal(a, b);
    }

    [Fact]
    public void TestConvertToBoolException()
    {
        var a = new bool[] { true, false, false, true };
        var b = new bool[a.Length - 1];
        var c = new BooleanConverters();
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, b, CastMode.KDefault));
    }

    [Fact]
    public void TestConvertToSbyte()
    {
        var a = new bool[] { true, false, false, true };
        var expected = a.Select(x => x ? (sbyte)1 : (sbyte)0).ToArray();
        var actual = new sbyte[a.Length];
        var c = new BooleanConverters();

        var modes = new CastMode[] { CastMode.KDefault, CastMode.CheckOverflow, CastMode.Reinterpret };
        foreach (var mode in modes)
        {
            Array.Clear(actual);
            Assert.NotEqual(expected, actual);

            c.ConvertTo(a, actual, CastMode.KDefault);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void TestConvertToSbyteException()
    {
        var a = new bool[] { true, false, false, true };
        var actual1 = new sbyte[a.Length];
        var actual2 = new sbyte[a.Length - 1];
        var c = new BooleanConverters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
    }

    [Fact]
    public void TestConvertToByte()
    {
        var a = new bool[] { true, false, false, true };
        var expected = a.Select(x => x ? (byte)1 : (byte)0).ToArray();
        var actual = new byte[a.Length];
        var c = new BooleanConverters();

        var modes = new CastMode[] { CastMode.KDefault, CastMode.CheckOverflow, CastMode.Reinterpret };
        foreach (var mode in modes)
        {
            Array.Clear(actual);
            Assert.NotEqual(expected, actual);

            c.ConvertTo(a, actual, CastMode.KDefault);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void TestConvertToByteException()
    {
        var a = new bool[] { true, false, false, true };
        var actual1 = new byte[a.Length];
        var actual2 = new byte[a.Length - 1];
        var c = new BooleanConverters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
    }

    [Fact]
    public void TestConvertToShort()
    {
        var a = new bool[] { true, false, false, true };
        var expected = a.Select(x => x ? (short)1 : (short)0).ToArray();
        var actual = new short[a.Length];
        var c = new BooleanConverters();

        var modes = new CastMode[] { CastMode.KDefault, CastMode.CheckOverflow, CastMode.Reinterpret };
        foreach (var mode in modes)
        {
            Array.Clear(actual);
            Assert.NotEqual(expected, actual);

            c.ConvertTo(a, actual, CastMode.KDefault);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void TestConvertToShortException()
    {
        var a = new bool[] { true, false, false, true };
        var actual1 = new short[a.Length];
        var actual2 = new short[a.Length - 1];
        var c = new BooleanConverters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
    }

    [Fact]
    public void TestConvertToUshort()
    {
        var a = new bool[] { true, false, false, true };
        var expected = a.Select(x => x ? (ushort)1 : (ushort)0).ToArray();
        var actual = new ushort[a.Length];
        var c = new BooleanConverters();

        var modes = new CastMode[] { CastMode.KDefault, CastMode.CheckOverflow, CastMode.Reinterpret };
        foreach (var mode in modes)
        {
            Array.Clear(actual);
            Assert.NotEqual(expected, actual);

            c.ConvertTo(a, actual, CastMode.KDefault);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void TestConvertToUshortException()
    {
        var a = new bool[] { true, false, false, true };
        var actual1 = new ushort[a.Length];
        var actual2 = new ushort[a.Length - 1];
        var c = new BooleanConverters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
    }

    [Fact]
    public void TestConvertToInt()
    {
        var a = new bool[] { true, false, false, true };
        var expected = a.Select(x => x ? (int)1 : (int)0).ToArray();
        var actual = new int[a.Length];
        var c = new BooleanConverters();

        var modes = new CastMode[] { CastMode.KDefault, CastMode.CheckOverflow, CastMode.Reinterpret };
        foreach (var mode in modes)
        {
            Array.Clear(actual);
            Assert.NotEqual(expected, actual);

            c.ConvertTo(a, actual, CastMode.KDefault);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void TestConvertToIntException()
    {
        var a = new bool[] { true, false, false, true };
        var actual1 = new int[a.Length];
        var actual2 = new int[a.Length - 1];
        var c = new BooleanConverters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
    }

    [Fact]
    public void TestConvertToUint()
    {
        var a = new bool[] { true, false, false, true };
        var expected = a.Select(x => x ? 1U : 0U).ToArray();
        var actual = new uint[a.Length];
        var c = new BooleanConverters();

        var modes = new CastMode[] { CastMode.KDefault, CastMode.CheckOverflow, CastMode.Reinterpret };
        foreach (var mode in modes)
        {
            Array.Clear(actual);
            Assert.NotEqual(expected, actual);

            c.ConvertTo(a, actual, CastMode.KDefault);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void TestConvertToUintException()
    {
        var a = new bool[] { true, false, false, true };
        var actual1 = new uint[a.Length];
        var actual2 = new uint[a.Length - 1];
        var c = new BooleanConverters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
    }

    [Fact]
    public void TestConvertToLong()
    {
        var a = new bool[] { true, false, false, true };
        var expected = a.Select(x => x ? 1L : 0L).ToArray();
        var actual = new long[a.Length];
        var c = new BooleanConverters();

        var modes = new CastMode[] { CastMode.KDefault, CastMode.CheckOverflow, CastMode.Reinterpret };
        foreach (var mode in modes)
        {
            Array.Clear(actual);
            Assert.NotEqual(expected, actual);

            c.ConvertTo(a, actual, CastMode.KDefault);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void TestConvertToLongException()
    {
        var a = new bool[] { true, false, false, true };
        var actual1 = new long[a.Length];
        var actual2 = new long[a.Length - 1];
        var c = new BooleanConverters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
    }

    [Fact]
    public void TestConvertToUlong()
    {
        var a = new bool[] { true, false, false, true };
        var expected = a.Select(x => x ? 1UL : 0UL).ToArray();
        var actual = new ulong[a.Length];
        var c = new BooleanConverters();

        var modes = new CastMode[] { CastMode.KDefault, CastMode.CheckOverflow, CastMode.Reinterpret };
        foreach (var mode in modes)
        {
            Array.Clear(actual);
            Assert.NotEqual(expected, actual);

            c.ConvertTo(a, actual, CastMode.KDefault);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void TestConvertToUlongException()
    {
        var a = new bool[] { true, false, false, true };
        var actual1 = new ulong[a.Length];
        var actual2 = new ulong[a.Length - 1];
        var c = new BooleanConverters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
    }

    [Fact]
    public void TestConvertToHalf()
    {
        var a = new bool[] { true, false, false, true };
        var expected = a.Select(x => x ? (Half)1F : (Half)0F).ToArray();
        var actual = new Half[a.Length];
        var c = new BooleanConverters();

        var modes = new CastMode[] { CastMode.KDefault, CastMode.CheckOverflow, CastMode.Reinterpret };
        foreach (var mode in modes)
        {
            Array.Clear(actual);
            Assert.NotEqual(expected, actual);

            c.ConvertTo(a, actual, CastMode.KDefault);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void TestConvertToHalfException()
    {
        var a = new bool[] { true, false, false, true };
        var actual1 = new Half[a.Length];
        var actual2 = new Half[a.Length - 1];
        var c = new BooleanConverters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
    }

    [Fact]
    public void TestConvertToFloat()
    {
        var a = new bool[] { true, false, false, true };
        var expected = a.Select(x => x ? 1F : 0F).ToArray();
        var actual = new float[a.Length];
        var c = new BooleanConverters();

        var modes = new CastMode[] { CastMode.KDefault, CastMode.CheckOverflow, CastMode.Reinterpret };
        foreach (var mode in modes)
        {
            Array.Clear(actual);
            Assert.NotEqual(expected, actual);

            c.ConvertTo(a, actual, CastMode.KDefault);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void TestConvertToFloatException()
    {
        var a = new bool[] { true, false, false, true };
        var actual1 = new float[a.Length];
        var actual2 = new float[a.Length - 1];
        var c = new BooleanConverters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
    }

    [Fact]
    public void TestConvertToDouble()
    {
        var a = new bool[] { true, false, false, true };
        var expected = a.Select(x => x ? 1D : 0D).ToArray();
        var actual = new double[a.Length];
        var c = new BooleanConverters();

        var modes = new CastMode[] { CastMode.KDefault, CastMode.CheckOverflow, CastMode.Reinterpret };
        foreach (var mode in modes)
        {
            Array.Clear(actual);
            Assert.NotEqual(expected, actual);

            c.ConvertTo(a, actual, CastMode.KDefault);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void TestConvertToDoubleException()
    {
        var a = new bool[] { true, false, false, true };
        var actual1 = new double[a.Length];
        var actual2 = new double[a.Length - 1];
        var c = new BooleanConverters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
    }

    [Fact]
    public void TestConvertToBFloat16()
    {
        var a = new bool[] { true, false, false, true };
        var expected = a.Select(x => x ? (BFloat16)1F : (BFloat16)0F).ToArray();
        var actual = new BFloat16[a.Length];
        var c = new BooleanConverters();

        var modes = new CastMode[] { CastMode.KDefault, CastMode.CheckOverflow, CastMode.Reinterpret };
        foreach (var mode in modes)
        {
            Array.Clear(actual);
            Assert.NotEqual(expected, actual);

            c.ConvertTo(a, actual, CastMode.KDefault);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void TestConvertToBFloat16Exception()
    {
        var a = new bool[] { true, false, false, true };
        var actual1 = new BFloat16[a.Length];
        var actual2 = new BFloat16[a.Length - 1];
        var c = new BooleanConverters();

        Assert.Throws<InvalidCastException>(() => c.ConvertTo(a, actual1, CastMode.Exact));
        Assert.Throws<ArgumentException>(() => c.ConvertTo(a, actual2, CastMode.KDefault));
    }
}
