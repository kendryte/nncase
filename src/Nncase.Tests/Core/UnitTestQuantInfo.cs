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
        var quantParameter = mixQuantInfo.QuantParameter;
        bool hasBindedMixQuantInfo = mixQuantInfo.HasBindedMixQuantInfo;
        var markerQuantType = mixQuantInfo.MarkerQuantType;
        bool doSquant = mixQuantInfo.DoSquant;
        var i8FineTunedWeights = mixQuantInfo.I8FineTunedWeights;
        var u8FineTunedWeights = mixQuantInfo.U8FineTunedWeights;
        var i8FineTunedWeightsRangesByChannel = mixQuantInfo.I8FineTunedWeightsRangesByChannel;
        var u8FineTunedWeightsRangesByChannel = mixQuantInfo.U8FineTunedWeightsRangesByChannel;
    }

    [Fact]
    public void TestAdaQuantInfo()
    {
        var adaQuantInfo = new AdaQuantInfo();
        var inputQuantParameter = adaQuantInfo.InputQuantParameter;
        var adaRoundRefTensor = adaQuantInfo.AdaRoundRefTensor;
    }
}
