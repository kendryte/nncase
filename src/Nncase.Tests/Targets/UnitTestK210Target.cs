// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#if false
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Quantization;
using Nncase.Runtime.Interop;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.Targets;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestK210Target : TestClassBase
{
    public UnitTestK210Target()
    {
        DefaultTargetName = "k210";
        CompileOptions.QuantizeOptions.ModelQuantMode = ModelQuantMode.UsePTQ;
    }

    [Fact]
    [AutoSetupTestMethod(InitSession = false)]
    public void TestCreateK210Target()
    {
        var target = CompilerServices.GetTarget("k210");
        Assert.NotNull(target);
    }

    [Theory]
    [CombinatorialData]
    public void TestCreateModuleBuilders([CombinatorialValues("stackvm", "kpu")] string moduleKind)
    {
        var moduleBuilder = CompileSession.Target.CreateModuleBuilder(moduleKind, CompileSession.CompileOptions);
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
        var y = IR.F.NN.Conv2D(
            x,
            w,
            b,
            new[] { 1, 1 },
            new[,]
            {
                { 0, 0 },
                { 0, 0 },
            },
            new[] { 1, 1 },
            PadMode.Constant,
            1);
        await TestCodeGenAsync(y, new[] { x });
    }

    private async Task TestCodeGenAsync(Expr body, Var[] vars)
    {
        var main = new Function("main", body, vars);
        var module = new IRModule(main);

        // 1. Optimize target dependent
        CompileOptions.QuantizeOptions.CalibrationDataset = new RandomCalibrationDatasetProvider(vars, 1);
        CompileOptions.QuantizeOptions.CalibrationMethod = CalibMethod.Kld;

        var pmgr = CompileSession.CreatePassManager("Passes");
        CompileSession.Target.RegisterTargetDependentPass(pmgr, CompileOptions);
        await pmgr.RunAsync(module);

        // var modelBuilder = new ModelBuilder(target);
        // var linkedModel = modelBuilder.Build(module);
        // using var output = File.Open($"k210_{name}/test.kmodel", FileMode.Create);
        // linkedModel.Serialize(output);
        // Assert.NotEqual(0, output.Length);
    }

    private void GenerateKModelAndRun(IRModule module, Tensor input, Tensor[] expectedOutput, [CallerMemberName] string? name = null)
    {
        var modelBuilder = CompileSession.GetRequiredService<IModelBuilder>();
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
        var rtOutput = entry!.Invoke(rtInput);
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
}
#endif
