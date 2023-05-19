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
    public async Task TestRandomCalibrationDatasetProvider1()
    {
        var vars = new List<Var> { new Var("x", DataTypes.Float32) };
        var samplesCount = 5;
        var provider = new RandomCalibrationDatasetProvider(vars, samplesCount);

        var sampleCount = provider.Count;
        var samples = await provider.Samples.ToListAsync();

        Assert.Equal(samplesCount, sampleCount);
        Assert.Equal(samplesCount, samples.Count);
    }

    [Fact]
    public async Task TestRandomCalibrationDatasetProvider2()
    {
        var vars = new List<Var>
        {
            new Var("x", DataTypes.Float32),
            new Var("y", DataTypes.Int32),
        };
        var samplesCount = 5;
        var provider = new RandomCalibrationDatasetProvider(vars, samplesCount);

        var samples = await provider.Samples.ToListAsync();

        foreach (var sample in samples)
        {
            Assert.True(sample.ContainsKey(vars[0]));
            Assert.True(sample[vars[0]].Type == DataTypes.Float32);

            Assert.True(sample.ContainsKey(vars[1]));
            Assert.True(sample[vars[1]].Type == DataTypes.Int32);
        }
    }
}
