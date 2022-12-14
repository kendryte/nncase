// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Compiler;
using Nncase.IR;
using Nncase.Quantization;
using Nncase.Transform;
using Nncase.Transform.Passes;
using Xunit;

namespace Nncase.Tests.CompilerTest;

internal sealed class RandCalibrationDatasetProvider : ICalibrationDatasetProvider
{

    private const int count = 5;
    public int? Count => count;

    public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples { get; }

    public RandCalibrationDatasetProvider(IEnumerable<Var> vars)
    {
        Samples = Enumerable.Range(0, count).Select(i =>
        {
            var values = new Dictionary<Var, IValue>();
            foreach (var var in vars)
            {
                CompilerServices.InferenceType(var);
                var shape = var.CheckedShape.Select(d => d.IsUnknown ? 1 : d.FixedValue).ToArray();
                var value = Value.FromTensor(IR.F.Random.Normal(var.CheckedDataType, 0, 1, 0, shape).Evaluate().AsTensor());
                values.Add(var, value);
            }
            return values;
        }).ToAsyncEnumerable();
    }
}

public class UnitTestCompiler : TestFixture.UnitTestFixtrue
{
    [Fact]
    public async Task TestTfliteNoptq()
    {
        var passOptions = GetPassOptions();
        var compileOptions = passOptions.CompileOptions;

        compileOptions.Target = "cpu";
        compileOptions.InputFile = Path.Combine(GetSolutionDirectory(), "tests/models/conv.tflite");
        compileOptions.InputFormat = "tflite";
        compileOptions.DumpLevel = 4;
        compileOptions.DumpDir = "UnitTestCompiler_tflite_noptq";
        compileOptions.OutputFile = Path.Combine(compileOptions.DumpDir, "test.kmodel");
        await Compile(compileOptions);
    }

    [Fact]
    public async Task TestTflitePtq()
    {
        var passOptions = GetPassOptions();
        var compileOptions = passOptions.CompileOptions;

        compileOptions.Target = "cpu";
        compileOptions.InputFile = Path.Combine(GetSolutionDirectory(), "tests/models/conv.tflite");
        compileOptions.InputFormat = "tflite";
        compileOptions.DumpLevel = 4;
        compileOptions.DumpDir = "UnitTestCompiler_tflite_ptq";
        compileOptions.OutputFile = Path.Combine(compileOptions.DumpDir, "test.kmodel");
        compileOptions.QuantType = DataTypes.UInt8;
        compileOptions.WQuantType = DataTypes.UInt8;
        compileOptions.ModelQuantMode = ModelQuantMode.UsePTQ;
        var in_shape = new[] { 1, 16, 16, 3 };
        Var input = new Var("serving_default_x:0", new TensorType(DataTypes.Float32, in_shape));
        compileOptions.QuantizeOptions = new()
        {
            CalibrationMethod = CalibMethod.Random,
            CalibrationDataset = new RandCalibrationDatasetProvider(new Var[] { input }),
        };

        await Compile(compileOptions);
    }

    [Fact]
    public async Task TestOnnxNoptq()
    {
        var passOptions = GetPassOptions();
        var compileOptions = passOptions.CompileOptions;

        compileOptions.Target = "cpu";
        compileOptions.InputFile = Path.Combine(GetSolutionDirectory(), "tests/models/conv.onnx");
        compileOptions.InputFormat = "onnx";
        compileOptions.DumpLevel = 4;
        compileOptions.DumpDir = "UnitTestCompiler_onnx_noptq";
        compileOptions.OutputFile = Path.Combine(compileOptions.DumpDir, "test.kmodel");
        await Compile(compileOptions);
    }

    [Fact]
    public async Task TestOnnxPtq()
    {
        var passOptions = GetPassOptions();
        var compileOptions = passOptions.CompileOptions;

        compileOptions.Target = "cpu";
        compileOptions.InputFile = Path.Combine(GetSolutionDirectory(), "tests/models/conv.onnx");
        compileOptions.InputFormat = "onnx";
        compileOptions.DumpLevel = 4;
        compileOptions.DumpDir = "UnitTestCompiler_onnx_ptq";
        compileOptions.OutputFile = Path.Combine(compileOptions.DumpDir, "test.kmodel");
        compileOptions.QuantType = DataTypes.UInt8;
        compileOptions.WQuantType = DataTypes.UInt8;
        compileOptions.ModelQuantMode = ModelQuantMode.UsePTQ;
        var in_shape = new[] { 1, 3, 16, 16 };
        Var input = new Var("input", new TensorType(DataTypes.Float32, in_shape));
        compileOptions.QuantizeOptions = new()
        {
            CalibrationMethod = CalibMethod.Random,
            CalibrationDataset = new RandCalibrationDatasetProvider(new Var[] { input }),
        };

        await Compile(compileOptions);
    }

    private async Task Compile(CompileOptions compileOptions)
    {
        // 1. create compiler
        var compiler = new Compiler.Compiler(compileOptions);

        // 2. import the model
        using (var model_stream = File.OpenRead(compileOptions.InputFile))
        {
            compiler.ImportModule(model_stream);
        }

        // 3. compile
        compiler.Compile();

        // 4. codegen
        using (var os = File.OpenWrite(compileOptions.OutputFile))
        {
            compiler.Gencode(os);
        }

        // check
        var stream = File.OpenRead(compileOptions.OutputFile);
        var bytes = new byte[stream.Length];
        stream.Read(bytes);
        Assert.True(bytes.Length > 0);
        Assert.True(bytes[0] == 0x4c);
        Assert.True(bytes[1] == 0x44);
        Assert.True(bytes[2] == 0x4d);
        Assert.True(bytes[3] == 0x4b);
    }
}