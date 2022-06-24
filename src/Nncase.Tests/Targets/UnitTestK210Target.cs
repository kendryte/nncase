using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.Runtime.Interop;
using Nncase.Transform;
using Xunit;

namespace Nncase.Tests.Targets;

public class UnitTestK210Target
{
    public UnitTestK210Target()
    {
        CompileOptions = new CompileOptions(true);
    }

    public ICompileOptions CompileOptions { get; }

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
        var moduleBuilder = target.CreateModuleBuilder("stackvm");
        Assert.NotNull(moduleBuilder);
    }

    [Fact]
    public void TestCreateKPUModuleBuilder()
    {
        var target = CompilerServices.GetTarget("k210");
        var moduleBuilder = target.CreateModuleBuilder("kpu");
        Assert.NotNull(moduleBuilder);
    }

    [Fact]
    public void TestSimpleConv2D()
    {
        var inChannels = 64;
        var outChannels = 8;
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1, inChannels, 4, 4 }));
        var w = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { outChannels, inChannels, 1, 1 }).Evaluate().AsTensor();
        var b = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { outChannels }).Evaluate().AsTensor();
        var y = IR.F.NN.Conv2D(x, w, b, new[] { 1, 1 }, new[,] { { 0, 0 }, { 0, 0 } }, new[] { 1, 1 }, PadMode.Constant, 1);
        TestCodeGen(y, new[] { x });
    }

    [Fact]
    public void TestCodeGenUseVarMultiTimes()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = x + 1.0f + x;
        TestCodeGen(y, new[] { x });
    }

    [Fact]
    public void TestCodeGenTuple()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = x + 1.0f + x;
        var z = y * 2.0f;
        TestCodeGen(new IR.Tuple(y, z), new[] { x });
    }

    private void TestCodeGen(Expr body, Var[] vars, [CallerMemberName] string name = null)
    {
        var main = new Function("main", body, vars);
        var module = new IRModule(main);
        var target = CompilerServices.GetTarget("k210");
        var dumpDir = "k210_" + name;
        var passOptions = new RunPassOptions(target, 2, dumpDir, CompileOptions);
        Directory.Delete(dumpDir, true);

        // 1. Optimize target dependent
        var pmgr = new PassManager(module, passOptions);
        target.RegisterTargetDependentPass(pmgr, CompileOptions);
        pmgr.Run();

        var modelBuilder = new ModelBuilder(target);
        var linkedModel = modelBuilder.Build(module);
        using var output = File.Open($"k210_{name}/test.kmodel", FileMode.Create);
        linkedModel.Serialize(output);
        Assert.NotEqual(0, output.Length);
    }

    [Fact]
    public void TestSimpleBinary()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = x + 1.0f;
        var main = new Function("main", y, new[] { x });
        var module = new IRModule(main);
        GenerateKModelAndRun(module, new[] { 1.0f }, new[] { 2.0f });
    }

    [Fact]
    public void TestTupleOutput()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var main = new Function("main", new IR.Tuple(x + 1.0f, x * 3.0f), new[] { x });
        var module = new IRModule(main);
        GenerateKModelAndRun(module, new[] { 1.0f }, new[] { (Tensor)2.0f, 3.0f });
    }

    [Fact]
    public void TestCallFunction()
    {
        var a = new Var("a");
        var b = a + 1.0f;
        var funcA = new Function("funcA", b, new[] { a });

        var x = new Var("x");
        var y = new Call(funcA, x + 1.0f);
        var main = new Function("main", y, new[] { x });
        var module = new IRModule(main);
        module.Add(funcA);
        GenerateKModelAndRun(module, new[] { 1.0f }, new[] { 3.0f });
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

        var interp = new RTInterpreter();
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
}
