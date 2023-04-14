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
    public Var[] Setup()
    {
        var input1 = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 224, 224 }));
        var input2 = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 224, 224 }));
        return new[] { input1, input2 };
    }

    [Fact]
    public async Task TestPytestCalibrationDatasetProvider()
    {
        var vars = Setup();
        var dataset = "./public";
        foreach (var t in vars)
        {
            var actual = IR.F.Random.Uniform(t.CheckedDataType, 1.0f, -1.0f, 0, t.CheckedShape).Evaluate().AsTensor();
            Tensors(new[] { actual }, dataset);
            var provider = new PytestCalibrationDatasetProvider(new[] { t }, dataset);
            Assert.Equal(1, provider.Count);
            var samples = provider.Samples;
            await foreach (var sample in samples)
            {
                Assert.Equal(sample[t].AsTensor(), actual);
            }
        }
    }

    private static void Tensors(Tensor[] tensorValue, string dir)
    {
        Directory.CreateDirectory(dir);
        foreach (var t in tensorValue)
        {
            var value = t;
            var sr1 = new StreamWriter(Path.Join(dir, "input_0_0.txt"));
            TensorTxt(value, sr1);
            var sr2 = Path.Join(dir, "input_0_0.bin");
            TensorBin(value, sr2);
        }
    }

    private static void TensorTxt(Tensor tensorValue, StreamWriter writer)
    {
        var tensor = tensorValue;
        var desc = "# (";
        for (var s = 0; s < tensor.Shape.Count; s++)
        {
            desc += tensor.Shape.ToValueArray()[s];
            if (s < tensor.Shape.Count - 1)
            {
                desc += " ,";
            }
        }

        desc += ")";
        writer.WriteLine(desc);
        foreach (var v in tensor.ToArray<float>())
        {
            writer.WriteLine(v);
        }
    }

    private static void TensorBin(Tensor tensorValue, string file)
    {
        File.WriteAllBytes(file, tensorValue.BytesBuffer.ToArray());
    }
}
