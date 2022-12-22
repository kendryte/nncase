﻿// Copyright (c) Canaan Inc. All rights reserved.
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
using Nncase.IR.Tensors;
using Nncase.Runtime.Interop;
using Xunit;
using static Nncase.IR.F.Tensors;
using GetItem = Nncase.IR.Tensors.GetItem;

namespace Nncase.Tests.Targets;

public class UnitTestCPUTarget
{
    public static IEnumerable<object[]> TestGetItemData =>
        new[]
        {
            new object[] { new[] { 0, 1 } },
            new object[] { new[] { 0, -1 } },
        };

    [Fact]
    public void TestCreateCPUTarget()
    {
        var target = CompilerServices.GetTarget("cpu");
        Assert.NotNull(target);
    }

    [Fact]
    public void TestCreateStackVMModuleBuilder()
    {
        var target = CompilerServices.GetTarget("cpu");
        var moduleBuilder = target.CreateModuleBuilder("stackvm", CompilerServices.CompileOptions);
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
    public void TestSimpleBinary()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = x + 1.0f;
        var main = new Function("main", y, new[] { x });
        var module = new IRModule(main);
        GenerateKModelAndRun(module, new[] { 1.0f }, new[] { 2.0f });
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

    private void TestCodeGen(Expr body, Var[] vars, [CallerMemberName] string name = null)
    {
        var main = new Function("main", body, vars);
        var module = new IRModule(main);
        var target = CompilerServices.GetTarget("cpu");
        var modelBuilder = new ModelBuilder(target);
        var linkedModel = modelBuilder.Build(module);
        using var output = File.Open($"{name}.kmodel", FileMode.Create);
        linkedModel.Serialize(output);
        Assert.NotEqual(0, output.Length);
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

    private void GenerateKModelAndRunFromFn(Function fn, Tensor input, Tensor expectedOutput, [CallerMemberName] string? name = null)
    {
        GenerateKModelAndRun(new IRModule(fn), input, new[] { expectedOutput }, name);
    }

    private void GenerateKModelAndRunFromFn(Function fn, Tensor input, Tensor[] expectedOutput, [CallerMemberName] string? name = null)
    {
        GenerateKModelAndRun(new IRModule(fn), input, expectedOutput, name);
    }
}
