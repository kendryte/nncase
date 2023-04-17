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
        foreach (var t in vars)
        {
            var actual = IR.F.Random.Uniform(t.CheckedDataType, 1.0f, -1.0f, 0, t.CheckedShape).Evaluate().AsTensor();
            DumpTensors(new[] { actual }, dataset, 2);
            var provider = new PytestCalibrationDatasetProvider(new[] { t }, dataset);
            Assert.Equal(2, provider.Count);
            var samples = provider.Samples;
            await foreach (var sample in samples)
            {
                Assert.Equal(sample[t].AsTensor(), actual);
            }
        }
    }

    [Fact]
    public async Task TestPytestCalibrationDatasetProvider2()
    {
        var vars = Setup();
        var dataset = "./public/test2";
        foreach (var t in vars)
        {
            var actual = IR.F.Random.Uniform(t.CheckedDataType, 1.0f, -1.0f, 0, t.CheckedShape).Evaluate().AsTensor();
            DumpTensors(new[] { actual }, dataset);
            var provider = new PytestCalibrationDatasetProvider(new[] { t }, dataset);
            Assert.Equal(1, provider.Count);
            var samples = provider.Samples;
            await foreach (var sample in samples)
            {
                Assert.Equal(sample[t].AsTensor(), actual);
            }
        }
    }

    [Fact]
    public async Task TestPytestCalibrationDatasetProvider3()
    {
        var vars1 = Setup();
        var dataset = "./public/test3";
        var actual1 = IR.F.Random.Uniform(vars1[0].CheckedDataType, 1.0f, -1.0f, 0, vars1[0].CheckedShape).Evaluate().AsTensor();
        var actual2 = IR.F.Random.Uniform(vars1[1].CheckedDataType, 1.0f, -1.0f, 0, vars1[1].CheckedShape).Evaluate().AsTensor();
        DumpTensors(new[] { actual1, actual2 }, dataset);
        var provider1 = new PytestCalibrationDatasetProvider(vars1, dataset);
        Assert.Equal(1, provider1.Count);
        var samples1 = provider1.Samples;
        await foreach (var sample in samples1)
        {
            Assert.Equal(sample[vars1[0]].AsTensor(), actual1);
            Assert.Equal(sample[vars1[1]].AsTensor(), actual2);
        }
    }

    [Fact]
    public async Task TestPytestCalibrationDatasetProvider4()
    {
        var vars1 = Setup();
        var dataset = "./public/test4";
        var actual1 = IR.F.Random.Uniform(vars1[0].CheckedDataType, 1.0f, -1.0f, 0, vars1[0].CheckedShape).Evaluate().AsTensor();
        var actual2 = IR.F.Random.Uniform(vars1[1].CheckedDataType, 1.0f, -1.0f, 0, vars1[1].CheckedShape).Evaluate().AsTensor();
        DumpTensors(new[] { actual1, actual2 }, dataset, 2);
        var provider1 = new PytestCalibrationDatasetProvider(vars1, dataset);
        Assert.Equal(2, provider1.Count);
        var samples1 = provider1.Samples;
        await foreach (var sample in samples1)
        {
            Assert.Equal(sample[vars1[0]].AsTensor(), actual1);
            Assert.Equal(sample[vars1[1]].AsTensor(), actual2);
        }
    }

    private static void DumpTensors(Tensor[] tensorValue, string dir, int sample = 1)
    {
        Directory.CreateDirectory(dir);
        for (var s = 0; s< sample; s++)
        {
            for (var t = 0; t < tensorValue.Length; t++)
            {
                var value = tensorValue[t];
                var sr1 = new StreamWriter(Path.Join(dir, $"input_{s}_{t}.txt"));
                DumpTxt(value, sr1);
                var sr2 = Path.Join(dir, $"input_{s}_{t}.bin");
                DumpBin(value, sr2);
            }
        }
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
