// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Text;
using Nncase;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestTensorHelper
{
    [Fact]
    public void TestToArray()
    {
        var a = new float[] { 1, 1, 1, 1, 1, 1, 1, 1 };
        var t = Tensor.Ones<float>(new int[] { 1, 1, 2, 4 });
        var b = t.ToArray<float>();
        Assert.Equal(a, b);
    }

    [Fact]
    public void TestToScalar1()
    {
        var t = Tensor.Ones<float>(new int[] { 1 });
        Assert.Equal(1F, t.ToScalar<float>());
    }

    [Fact]
    public void TestToScalar2()
    {
        var t = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        Assert.Throws<InvalidOperationException>(() => t.ToScalar<float>());
    }

    [Fact]
    public void TestToStr1()
    {
        var utf8 = new UTF8Encoding();
        string expected = "hello, world!";
        var bytes = utf8.GetBytes(expected);

        var t = Tensor.FromBytes(DataTypes.Utf8Char, new Memory<byte>(bytes), new int[] { bytes.Length });
        Assert.Equal(expected, t.ToStr());
    }

    [Fact]
    public void TestToStr2()
    {
        var t = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        Assert.Throws<InvalidCastException>(() => t.ToStr());
    }
}
