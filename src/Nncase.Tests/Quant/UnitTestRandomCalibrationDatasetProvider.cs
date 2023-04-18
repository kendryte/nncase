// Copyright (c) Canaan Inc. All rights reserved.
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

public class UnitTestRandomCalibrationDatasetProvider
{
    [Fact]
    public void TestRandomCalibrationDatasetProvider()
    {
        var input = new Var();
        var vars = new[] { input, input, input };
        var provider = new RandomCalibrationDatasetProvider(vars, 1000);
        Assert.Equal(1000, provider.Count);
        _ = provider.Samples;
    }
}
