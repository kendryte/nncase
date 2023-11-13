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
        var input2 = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 112, 224 }));
        return new[] { input1, input2 };
    }

    [Fact]
    public async Task TestPytestCalibrationDatasetProvider1()
    {
        var vars = Setup();
        var dataset = "./public/test1";
        var actuals = DumpTensors(dataset, vars, 2);
        var provider = new PytestCalibrationDatasetProvider(vars, dataset);
        Assert.Equal(2, provider.Count);
        var samples = provider.Samples;
        var count = 0;
        await foreach (var sample in samples)
        {
            Assert.Equal(sample[vars[0]].AsTensor(), actuals[count, 0]);
            Assert.Equal(sample[vars[1]].AsTensor(), actuals[count, 1]);
            count++;
        }
    }

    private static Tensor[,] DumpTensors(string dir, Var[] inputs, int sample)
    {
        Directory.CreateDirectory(dir);
        var outputs = new Tensor[sample, inputs.Length];
        for (var s = 0; s < sample; s++)
        {
            for (var t = 0; t < inputs.Length; t++)
            {
                var value = IR.F.Random.Uniform(inputs[t].CheckedDataType, 1.0f, -1.0f, s + t, inputs[t].CheckedShape).Evaluate().AsTensor();
                var sr1 = new StreamWriter(Path.Join(dir, $"input_{t}_{s}.txt"));
                DumpTxt(value, sr1);
                var sr2 = Path.Join(dir, $"input_{t}_{s}.bin");
                DumpBin(value, sr2);
                outputs[s, t] = value;
            }
        }

        return outputs;
    }

    private static void DumpTxt(Tensor tensorValue, StreamWriter writer)
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

    private static void DumpBin(Tensor tensorValue, string file)
    {
        File.WriteAllBytes(file, tensorValue.BytesBuffer.ToArray());
    }
}
