// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.Quantization;
using Nncase.Runtime.Interop;
using Nncase.Transform;
using Xunit;

namespace Nncase.Tests.Targets;

public class UnitTestK210Target
{
    public UnitTestK210Target()
    {
        CompileOptions = new CompileOptions(ModelQuantMode.UsePTQ);
    }

    public CompileOptions CompileOptions { get; }

    [Fact]
    public void TestCreateK210Target()
    {
        var target = CompilerServices.GetTarget("k210");
        Assert.NotNull(target);
    }

    [Fact]
    public void TestCreateStackVMModuleBuilder()
    {
        var target = CompilerServices.GetTarget("k210");
        var moduleBuilder = target.CreateModuleBuilder("stackvm", CompilerServices.CompileOptions);
        Assert.NotNull(moduleBuilder);
    }

    [Fact]
    public void TestCreateKPUModuleBuilder()
    {
        var target = CompilerServices.GetTarget("k210");
        var moduleBuilder = target.CreateModuleBuilder("kpu", CompilerServices.CompileOptions);
        Assert.NotNull(moduleBuilder);
    }

    [Fact]
    public async Task TestSimpleConv2D()
    {
        var inChannels = 64;
        var outChannels = 8;
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1, inChannels, 4, 4 }));
        var w = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { outChannels, inChannels, 1, 1 }).Evaluate().AsTensor();
        var b = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { outChannels }).Evaluate().AsTensor();
        var y = IR.F.NN.Conv2D(x, w, b, new[] { 1, 1 }, new[,]
        {
            { 0, 0 },
            { 0, 0 },
        }, new[] { 1, 1 }, PadMode.Constant, 1);
        await TestCodeGen(y, new[] { x });
    }

    private async Task TestCodeGen(Expr body, Var[] vars, [CallerMemberName] string? name = null)
    {
        var main = new Function("main", body, vars);
        var module = new IRModule(main);
        var target = CompilerServices.GetTarget("k210");
        var dumpDir = "k210_" + name;
        var passOptions = new RunPassOptions(target, 2, dumpDir, CompileOptions);
        if (Directory.Exists(dumpDir))
        {
            Directory.Delete(dumpDir, true);
        }

        // 1. Optimize target dependent
        CompileOptions.QuantizeOptions = new QuantizeOptions { CalibrationDataset = new RandomCalibrationDatasetProvider(vars), CalibrationMethod = CalibMethod.Kld };
        var pmgr = new PassManager(module, passOptions);
        target.RegisterTargetDependentPass(pmgr, CompileOptions);
        await pmgr.RunAsync();

        // var modelBuilder = new ModelBuilder(target);
        // var linkedModel = modelBuilder.Build(module);
        // using var output = File.Open($"k210_{name}/test.kmodel", FileMode.Create);
        // linkedModel.Serialize(output);
        // Assert.NotEqual(0, output.Length);
    }

    private void GenerateKModelAndRun(IRModule module, Tensor input, Tensor[] expectedOutput, [CallerMemberName] string? name = null)
    {
        var target = CompilerServices.GetTarget("cpu");
        var modelBuilder = new ModelBuilder(target);
        var linkedModel = modelBuilder.Build(module);
        using (var output = File.Open($"{name}.kmodel", FileMode.Create))
        {
            linkedModel.Serialize(output);
            Assert.NotEqual(0, output.Length);
        }

        byte[] kmodel;
        using (var output = new MemoryStream())
        {
            linkedModel.Serialize(output);
            kmodel = output.ToArray();
        }

        var interp = RTInterpreter.Create();
        interp.LoadModel(kmodel);
        var entry = interp.Entry;

        var rtInput = RTTensor.FromTensor(input);
        var rtOutput = entry.Invoke(rtInput);
        var rtOutputs = rtOutput is RTTensor t ? new[] { t } : ((RTTuple)rtOutput).Fields.Cast<RTTensor>().ToArray();
        Assert.Equal(expectedOutput.Length, rtOutputs.Length);

        for (int i = 0; i < rtOutputs.Length; i++)
        {
            var outBuffer = rtOutputs[i].Buffer.Buffer.AsHost()!;
            using (var mmOwner = outBuffer.Map(RTMapAccess.Read))
            {
                Assert.Equal(expectedOutput[i].BytesBuffer.ToArray(), mmOwner.Memory.Span.ToArray());
            }
        }
    }

    private void GenerateKModelAndRun(IRModule module, Tensor input, Tensor expectedOutput, [CallerMemberName] string? name = null)
    {
        GenerateKModelAndRun(module, input, new[] { expectedOutput }, name);
    }

    private class RandomCalibrationDatasetProvider : ICalibrationDatasetProvider
    {
        public RandomCalibrationDatasetProvider(IReadOnlyList<Var> vars)
        {
            var values = new Dictionary<Var, IValue>();
            foreach (var var in vars)
            {
                CompilerServices.InferenceType(var);
                var value = IR.F.Random.Normal(var.CheckedDataType, var.CheckedShape).Evaluate();
                values.Add(var, value);
            }

            Samples = new[] { values }.ToAsyncEnumerable();
        }

        public int? Count => 1;

        public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples { get; }
    }
}
