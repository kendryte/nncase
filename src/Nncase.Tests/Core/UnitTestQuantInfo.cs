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
        Assert.Equal(quantParameter, new List<QuantParam>());

        var hasBindedMixQuantInfo = mixQuantInfo.HasBindedMixQuantInfo;
        Assert.False(hasBindedMixQuantInfo);

        var markerQuantType = mixQuantInfo.MarkerQuantType;
        Assert.Equal(markerQuantType, DataTypes.Float32);

        var doSquant = mixQuantInfo.DoSquant;
        Assert.False(doSquant);

        var i8FineTunedWeights = mixQuantInfo.I8FineTunedWeights;
        Assert.Null(i8FineTunedWeights);

        var u8FineTunedWeights = mixQuantInfo.U8FineTunedWeights;
        Assert.Null(u8FineTunedWeights);

        var i8FineTunedWeightsRangesByChannel = mixQuantInfo.I8FineTunedWeightsRangesByChannel;
        Assert.Null(i8FineTunedWeightsRangesByChannel);

        var u8FineTunedWeightsRangesByChannel = mixQuantInfo.U8FineTunedWeightsRangesByChannel;
        Assert.Null(u8FineTunedWeightsRangesByChannel);
    }

    [Fact]
    public void TestAdaQuantInfo()
    {
        var adaQuantInfo = new AdaQuantInfo();
        var inputQuantParameter = adaQuantInfo.InputQuantParameter;
        Assert.Equal(inputQuantParameter, new QuantParam(0, 1.0f));

        var adaRoundRefTensor = adaQuantInfo.AdaRoundRefTensor;
        Assert.Null(adaRoundRefTensor);
    }
}
