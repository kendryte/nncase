// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Quantization;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using Xunit;
using static Nncase.IR.F.NN;
using Utility = Nncase.Quantization.Utility;

namespace Nncase.Tests.QuantTest;

public class UnitTestPytestCalibrationDatasetProvider
{
    [Fact]
    public async Task TestPytestCalibrationDatasetProvider()
    {
        var vars = new[] { new Var("0.090337") };
        var dataset = "./public";
        Tensors(new TensorValue[] { new TensorValue(0.090337) }, dataset);
        var provider = new PytestCalibrationDatasetProvider(vars, dataset);
        Assert.Equal(0, provider.Count);
        var samples = provider.Samples;
        await foreach (var sample in samples)
        {
            Assert.Equal(sample[vars[0]], new TensorValue(0.152153));
        }
    }

    private static void Tensors(TensorValue[] tensorValue, string dir)
    {
        Directory.CreateDirectory(dir);
        foreach (var t in tensorValue)
        {
            using var sr = new StreamWriter(Path.Join(dir, "input_0_0.txt"));
            Tensor(t, sr);
        }
    }

    private static void Tensor(TensorValue tensorValue, StreamWriter writer)
    {
        var tensor = tensorValue.AsTensor();
        writer.WriteLine("#" + tensor.Shape.ToArray());
        var number = tensor.ToArray<float>();
        writer.WriteLine($"${number[0]}");
    }
}
