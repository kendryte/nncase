﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Quantization;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.NN;
using Utility = Nncase.Quantization.Utility;

namespace Nncase.Tests.QuantTest;

public class UnitTestUtility
{
    [Fact]
    public void TestGetCosineSimilarity()
    {
        var v1 = new float[] { 1f, 2f, 3f, 4f };
        var v2 = new float[] { 1f };
        var actual = Utility.GetCosineSimilarity(v1, v2);
        Assert.Equal(1f, actual);
        Assert.Equal(1f, Utility.GetCosineSimilarity(new[] { 0f }, new[] { 0f }));
    }

    [Fact]
    public void TestGetFixedMul()
    {
        var actual1 = Utility.GetFixedMul(0f, 0, 0, true);
        var actual2 = Utility.GetFixedMul(2f, 1, 2, true);
        var actual3 = Utility.GetFixedMul(1f, 2, 1, true);
        var actual4 = Utility.GetFixedMul(1f, 1, 0, false);
        Assert.Equal(new FixedMul(0f, 0), actual1);
        Assert.Equal(new FixedMul(0f, 2), actual2);
        Assert.Equal(new FixedMul(0f, 1), actual3);
        Assert.Equal(new FixedMul(0f, 0), actual4);
        Assert.Throws<InvalidOperationException>(() => Utility.GetFixedMul(-1f, 0, 0, false));
        Assert.Throws<ArgumentOutOfRangeException>(() => Utility.GetFixedMul(100f, 0, 0, true));
    }
}
