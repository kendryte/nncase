// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestValueRange
{
    [Fact]
    public void TestFull()
    {
        {
            var full = ValueRange<byte>.Full;
            Assert.Equal(byte.MinValue, full.Min);
            Assert.Equal(byte.MaxValue, full.Max);
        }

        {
            var full = ValueRange<float>.Full;
            Assert.Equal(float.NegativeInfinity, full.Min);
            Assert.Equal(float.PositiveInfinity, full.Max);
        }

        {
            var full = ValueRange<Half>.Full;
            Assert.Equal(Half.NegativeInfinity, full.Min);
            Assert.Equal(Half.PositiveInfinity, full.Max);
        }

        {
            var full = ValueRange<BFloat16>.Full;
            Assert.Equal(BFloat16.NegInfinity, full.Min);
            Assert.Equal(BFloat16.Infinity, full.Max);
        }
    }

    [Fact]
    public void TestIsFull()
    {
        {
            var a = new ValueRange<byte>(1, 100);
            Assert.False(a.IsFull);

            var b = new ValueRange<byte>(byte.MinValue, byte.MaxValue);
            Assert.True(b.IsFull);
        }

        {
            var a = new ValueRange<float>(1.0F, 100.0F);
            Assert.False(a.IsFull);

            var b = new ValueRange<float>(float.NegativeInfinity, float.PositiveInfinity);
            Assert.True(b.IsFull);
        }

        {
            var a = new ValueRange<Half>((Half)1.0F, (Half)100.0F);
            Assert.False(a.IsFull);

            var b = new ValueRange<Half>(Half.NegativeInfinity, Half.PositiveInfinity);
            Assert.True(b.IsFull);
        }

        {
            var a = new ValueRange<BFloat16>((BFloat16)1.0F, (BFloat16)100.0F);
            Assert.False(a.IsFull);

            var b = new ValueRange<BFloat16>(BFloat16.NegInfinity, BFloat16.Infinity);
            Assert.True(b.IsFull);
        }
    }

    [Fact]
    public void TestUnion()
    {
        var a = new ValueRange<byte>(1, 100);
        var b = new ValueRange<byte>(0, 90);
        var c = a.Union(b);
        Assert.Equal(0, c.Min);
        Assert.Equal(100, c.Max);
    }
}
