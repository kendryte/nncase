// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase;
using Nncase.IR;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestQuantUtility
{
    [Fact]
    public void TestGetQuantParam()
    {
        var quantParam1 = QuantUtility.GetQuantParam(new ValueRange<float>(-1f, 1f), 8, QuantMode.SignedSymmetricMode);
        Assert.Equal(new QuantParam(0, 0.007874016f), quantParam1);

        var quantParam2 = QuantUtility.GetQuantParam(new ValueRange<float>(-1f, 1f), 8, QuantMode.SignedAsymmetricMode);
        Assert.Equal(new QuantParam(0, 0.007843138f), quantParam2);
    }

    [Fact]
    public void TestFixupRange()
    {
        var range1 = QuantUtility.FixupRange(new ValueRange<float>(0f, 0f));
        Assert.Equal(new ValueRange<float>(0f, 0.1f), range1);

        var range2 = QuantUtility.FixupRange(new ValueRange<float>(0f, 0f), true);
        Assert.Equal(new ValueRange<float>(-0.01f, 0.01f), range2);

        var range3 = QuantUtility.FixupRange(new ValueRange<float>(0f, 0.005f));
        Assert.Equal(new ValueRange<float>(0f, 0.01f), range3);
    }

    [Fact]
    public void TestGetRange()
    {
        var input = new[] { 1, 2, 3, 4 };
        Assert.Equal(new ValueRange<int>(1, 4), QuantUtility.GetRange<int>(input));
    }
}
