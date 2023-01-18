// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestTensorOfTHelper
{
    [Fact]
    public void TestToArray()
    {
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var tensor = Tensor.From(a, new int[] { 1, 1, 2, 4 });
        var b = tensor.ToArray();
        Assert.Equal(a, b);
    }

    [Fact]
    public void TestToScalar()
    {
        var scalar = 1F;

        var t1 = (Tensor<float>)scalar;
        Assert.Equal(scalar, t1.ToScalar());

        var t2 = new Tensor<float>(new int[] { 1, 3, 16, 16 });
        Assert.Throws<InvalidOperationException>(() => t2.ToScalar());
    }
}
