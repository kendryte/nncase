// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Nncase.CodeGen;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using Nncase.Runtime.Interop;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.Utilities;
using Xunit;
using static Nncase.IR.F.Tensors;
using GetItem = Nncase.IR.Tensors.GetItem;

namespace Nncase.Tests.Targets;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestCPUTarget : TestClassBase
{
    public UnitTestCPUTarget()
    {
        DefaultTargetName = CPUTarget.Kind;
#if DEBUG
        CompileOptions.DumpFlags = DumpFlags.PassIR | DumpFlags.Rewrite | DumpFlags.EGraphCost | DumpFlags.CodeGen;
#else
        CompileOptions.DumpFlags = DumpFlags.CodeGen;
#endif
    }

    public static IEnumerable<object[]> TestGetItemData =>
        new[]
        {
            new object[] { new[] { 0, 1 } },
            new object[] { new[] { 0, -1 } },
        };

    public static IEnumerable<object[]> TestIfData =>
        new[]
        {
            new object[] { true },
            new object[] { false },
        };

    [Fact]
    [AutoSetupTestMethod(InitSession = false)]
    public void TestCPUTargetKind()
    {
        Assert.Equal("cpu", CPUTarget.Kind);
    }

    [Fact]
    [AutoSetupTestMethod(InitSession = false)]
    public void TestCreateCPUTarget()
    {
        var target = CompilerServices.GetTarget(CPUTarget.Kind);
        Assert.NotNull(target);
    }

    [Theory]
    [CombinatorialData]
    public void TestCreateStackVMModuleBuilder([CombinatorialValues("stackvm")] string moduleKind)
    {
        var moduleBuilder = CompileSession.Target.CreateModuleBuilder(moduleKind, CompileOptions);
        Assert.NotNull(moduleBuilder);
    }

    [Fact]
    public void TestSimpleCodeGen()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = x + 1.0f;
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

    [Fact]
    public void TestCodeGenVisitLeafVar()
    {
        var main = new Function(new Var(), Array.Empty<Var>());
        var module = new IRModule(main);
        var modelBuilder = CompileSession.GetRequiredService<IModelBuilder>();
        Assert.Throws<InvalidOperationException>(() => modelBuilder.Build(module));
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
    public void TestSimpleUnary()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = IR.F.Math.Abs(x);
        var main = new Function("main", y, new[] { x });
        var module = new IRModule(main);
        GenerateKModelAndRun(module, new[] { -1.0f }, new[] { 1.0f });
    }

    [Fact]
    public void TestCodegenCallParamOrder()
    {
        // order is true: x - 3 = 2 - 3 = -1
        // order is false: 3 - x = 3 - 2 = 1
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = x - 3f;
        var main = new Function("main", y, new[] { x });
        GenerateKModelAndRunFromFn(main, new[] { 2f }, (Tensor)new[] { -1f });
    }

    [Fact]
    public void TestSimpleTupleOutput()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var main = new Function("main", new IR.Tuple(x + 1.0f, x * 3.0f), new[] { x });
        var module = new IRModule(main);
        GenerateKModelAndRun(module, new[] { 1.0f }, new[] { (Tensor)2.0f, 3.0f });
    }

    [Fact]
    public void TestTupleOrder()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var main = new Function("main", new IR.Tuple(x + 1.0f, x + 2f, x + 3f), new[] { x });
        GenerateKModelAndRunFromFn(main, new[] { 1f }, new[] { (Tensor)2f, 3f, 4f });
    }

    [Theory]
    [MemberData(nameof(TestGetItemData))]
    public void TestGetItem(int[] index)
    {
        var input = Tensor.From(new[] { 1, 2, 3, 4, 5, 6 }, new[] { 1, 2, 3 });
        var x = new Var("x", new TensorType(DataTypes.Int32, new[] { 1, 2, 3 }));
        var second = GetItem(x, index);
        var main = new Function("main", second, new[] { x });
        var dict = new Dictionary<Var, IValue>() { { x, Value.FromTensor(input) } };
        GenerateKModelAndRunFromFn(main, input, second.Evaluate(dict).AsTensor());
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

    [Theory(Skip = "CPU codegen currently doesn't support If")]
    [MemberData(nameof(TestIfData))]
    public void TestIf(bool input)
    {
        var condVar = new Var(new TensorType(DataTypes.Boolean, Shape.Scalar));
        var then = IR.F.Math.Abs(3f);
        var @else = IR.F.NN.Relu(Cast(3, DataTypes.Float32));
        var @if = IR.F.Math.Abs(new If(condVar, then, @else));

        Assert.True(@if.InferenceType());
        var main = new Function("main", @if, new[] { condVar });

        var output = @if.Evaluate(new Dictionary<Var, IValue> { { condVar, Value.FromTensor(input) } }).AsTensor();
        GenerateKModelAndRunFromFn(main, input, output);
    }

    [Fact(Skip = "If")]
    public void TestStackVMNestIf()
    {
        var condVar = new Var(new TensorType(DataTypes.Boolean, Shape.Scalar));
        _ = (Expr)3 - 1;
        var @else = (Expr)3 + 1;
        var elseThen = (Expr)8 * 8;
        var elsif = new If(condVar, elseThen, @else);

        var main = new Function("main", 2 * elsif, new[] { condVar });

        var input = (Tensor)true;
        var output = (Tensor)128;
        GenerateKModelAndRunFromFn(main, input, output);
    }

    [Fact(Skip="NotSupportIf")]
    public void TestNestIfWithThenBegin()
    {
        var condVar = new Var(new TensorType(DataTypes.Boolean, Shape.Scalar));
        var cast = Cast(condVar, DataTypes.Int32);
        var i = new If(condVar, cast * new If(condVar, 3 + cast, 2), 6);
        var main = new Function("main", i, new[] { condVar });
        Dumpper.DumpIR(main, "main");
        var input = (Tensor)true;
        var output = (Tensor)4;
        GenerateKModelAndRunFromFn(main, input, output);
    }

    [Fact(Skip = "NotSupportIf")]
    public void TestNestIfWithElseBegin()
    {
        var condVar = new Var(new TensorType(DataTypes.Boolean, Shape.Scalar));
        var i = new If(condVar, 3, new If(condVar, 1, 2));
        var main = new Function("main", i, new[] { condVar });
        var input = (Tensor)false;
        var output = (Tensor)2;
        GenerateKModelAndRunFromFn(main, input, output);
    }

    private void TestCodeGen(Expr body, Var[] vars, [CallerMemberName] string? name = null)
    {
        var main = new Function("main", body, vars);
        var module = new IRModule(main);
        var modelBuilder = CompileSession.GetRequiredService<IModelBuilder>();
        var linkedModel = modelBuilder.Build(module);
        using var output = File.Open($"{name}.kmodel", FileMode.Create);
        linkedModel.Serialize(output);
        Assert.NotEqual(0, output.Length);
    }

    private void GenerateKModelAndRun(IRModule module, Tensor input, Tensor[] expectedOutput, [CallerMemberName] string? name = null)
    {
        CompileSession.Compiler.ImportIRModule(module);
        CompileSession.Compiler.CompileAsync().Wait();

        byte[] kmodel;
        using (var output = new MemoryStream())
        {
            CompileSession.Compiler.Gencode(output);
            kmodel = output.ToArray();
        }

        if (Dumpper.IsEnabled(DumpFlags.CodeGen))
        {
            using (var kmodelFile = Dumpper.OpenFile($"{name}.kmodel"))
            {
                kmodelFile.Write(kmodel);
            }
        }

        var interp = RTInterpreter.Create();
        interp.LoadModel(kmodel);
        var entry = interp.Entry;
        Assert.NotNull(entry);

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

    private void GenerateKModelAndRunFromFn(Function fn, Tensor input, Tensor expectedOutput, [CallerMemberName] string? name = null)
    {
        GenerateKModelAndRun(new IRModule(fn), input, new[] { expectedOutput }, name);
    }

    private void GenerateKModelAndRunFromFn(Function fn, Tensor input, Tensor[] expectedOutput, [CallerMemberName] string? name = null)
    {
        GenerateKModelAndRun(new IRModule(fn), input, expectedOutput, name);
    }
}
