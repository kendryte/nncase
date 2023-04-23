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

public sealed class UnitTestQuantInfo
{
    [Fact]
    public void TestMixQuantInfo()
    {
        var mixQuantInfo = new MixQuantInfo();
        _ = mixQuantInfo.QuantParameter;
        _ = mixQuantInfo.HasBindedMixQuantInfo;
        _ = mixQuantInfo.MarkerQuantType;
        _ = mixQuantInfo.DoSquant;
        _ = mixQuantInfo.I8FineTunedWeights;
        _ = mixQuantInfo.U8FineTunedWeights;
        _ = mixQuantInfo.I8FineTunedWeightsRangesByChannel;
        _ = mixQuantInfo.U8FineTunedWeightsRangesByChannel;
    }

    [Fact]
    public void TestAdaQuantInfo()
    {
        var adaQuantInfo = new AdaQuantInfo();
        _ = adaQuantInfo.InputQuantParameter;
        _ = adaQuantInfo.AdaRoundRefTensor;
    }
}
