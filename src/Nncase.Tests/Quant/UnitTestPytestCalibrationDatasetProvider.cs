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
using Xunit;
using static Nncase.IR.F.NN;
using Utility = Nncase.Quantization.Utility;

namespace Nncase.Tests.QuantTest;

public class UnitTestPytestCalibrationDatasetProvider
{
    [Fact]
    public void TestPytestCalibrationDatasetProvider()
    {
        var vars = new[] { new Var("123"), new Var("234") };
        var dataset = "./public";
        Tensors(new TensorValue[] { new TensorValue(123), new TensorValue(234) }, dataset);
        var provider = new PytestCalibrationDatasetProvider(vars, dataset);
        Assert.Equal(0, provider.Count);
        _ = provider.Samples;
    }

    private static void Tensors(TensorValue[] tensorValue, string dir)
    {
        Directory.CreateDirectory(dir);
        for (var i = 0; i < tensorValue.Length; i++)
        {
            using (var sr = new StreamWriter(Path.Join(dir, "{i}.txt")))
            {
                Tensor(tensorValue[i], sr);
            }
        }
    }

    private static void Tensor(TensorValue tensorValue, StreamWriter writer)
    {
        var tensor = tensorValue.AsTensor();
        if (tensor.ElementType is PrimType)
        {
            var typeCode = ((PrimType)tensor.ElementType).TypeCode;
            writer.WriteLine($"type:{(int)typeCode}");
        }
        else
        {
            writer.WriteLine($"type:0");
        }

        writer.WriteLine(tensor.Shape.ToArray());

        var dt = tensor.ElementType;
        if (dt == DataTypes.Int8 || dt == DataTypes.Int32 || dt == DataTypes.Int64)
        {
            foreach (var v in tensor.ToArray<long>())
            {
                writer.WriteLine(v);
            }
        }
        else if (dt is PrimType)
        {
            foreach (var v in tensor.ToArray<float>())
            {
                writer.WriteLine(v);
            }
        }
        else
        {
            writer.WriteLine($"{dt} NotImpl");
        }
    }
}
