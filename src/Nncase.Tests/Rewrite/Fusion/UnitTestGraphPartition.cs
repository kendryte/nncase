// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Data.SqlTypes;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestPlatform.ObjectModel;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.Passes;
using Nncase.Passes.Analysis;
using Nncase.Passes.Rules.CPU;
using Nncase.PatternMatch;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.ReWrite.FusionTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestGraphPartition : TestClassBase
{
    public UnitTestGraphPartition()
    {
#if DEBUG
        CompileOptions.DumpFlags = DumpFlags.PassIR | DumpFlags.EGraphCost | DumpFlags.Rewrite;
#endif
    }

    [Fact]
    public async Task TestLineSameModuleI()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var main = new Function("main", IR.F.Math.Unary(UnaryOp.Abs, IR.F.Math.Unary(UnaryOp.Sin, input)), input);

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Unary>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallOp<Unary>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(2, pre_number);
        Assert.Equal(2, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestLineSmaeModuleC()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var main = new Function("main", IR.F.Math.Abs(IR.F.CPU.Boxing(input, new DistributedType(input.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")))), input);

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Unary>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(1, pre_number);
        Assert.Equal(1, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestLineDiffModuleC2I()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var main = new Function("main", IR.F.Math.Abs(IR.F.CPU.Boxing(input, input.CheckedType)), input);

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Unary>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(1, pre_number);
        Assert.Equal(1, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestLineDiffModuleI2C()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var main = new Function("main", IR.F.CPU.Boxing(IR.F.Math.Abs(input), input.CheckedType), input);

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Unary>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(1, pre_number);
        Assert.Equal(1, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestYSameModuleI()
    {
        var input1 = new Var("input1", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var input2 = new Var("input2", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var v_0 = IR.F.Math.Unary(UnaryOp.Cos, input1);
        var v_1 = IR.F.Math.Unary(UnaryOp.Neg, input2);
        var v_2 = IR.F.Math.Binary(BinaryOp.Add, v_0, v_1);
        var main = new Function("main", v_2, input1, input2);

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Unary>();

        var input_tensor1 = Testing.Rand<float>(1, 32, 32);
        var input_tensor2 = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input1, Value.FromTensor(input_tensor1) },
            { input2, Value.FromTensor(input_tensor2) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();
        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallOp<Unary>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(2, pre_number);
        Assert.Equal(2, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestYSameModuleC()
    {
        var input1 = new Var("input1", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var input2 = new Var("input2", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var v_0 = IR.F.CPU.Boxing(input1, new DistributedType(input1.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")));
        var v_1 = IR.F.CPU.Boxing(input2, new DistributedType(input2.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")));
        var v_2 = IR.F.Math.Binary(BinaryOp.Add, v_0, v_1);
        var main = new Function("main", v_2, input1, input2);

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Binary>();

        var input_tensor1 = Testing.Rand<float>(1, 32, 32);
        var input_tensor2 = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input1, Value.FromTensor(input_tensor1) },
            { input2, Value.FromTensor(input_tensor2) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();
        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallOp<Binary>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(1, pre_number);
        Assert.Equal(0, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestHandInHandSameModuleI()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var v_0 = IR.F.Math.Unary(UnaryOp.Abs, input);
        var v_1 = IR.F.Math.Binary(BinaryOp.Add, v_0, v_0);
        var main = new Function("main", v_1, input);
        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Unary>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();
        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(1, pre_number);
        Assert.Equal(0, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestHandInHandSameModuleC()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var v_0 = IR.F.CPU.Boxing(input, new DistributedType(input.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")));
        var v_1 = IR.F.Math.Binary(BinaryOp.Add, v_0, v_0);
        var main = new Function("main", v_1, input);
        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Boxing>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();
        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallOp<Boxing>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(1, pre_number);
        Assert.Equal(0, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestCircle1SameModule()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var v_0 = IR.F.CPU.Boxing(input, new DistributedType(input.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")));
        var v_1 = IR.F.Math.Unary(UnaryOp.Cos, v_0);
        var v_2 = IR.F.Math.Binary(BinaryOp.Add, v_0, v_1);

        var main = new Function("main", v_2, input);

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Unary>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();
        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(1, pre_number);
        Assert.Equal(1, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestCircle2SameModule()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var v_0 = IR.F.CPU.Boxing(input, new DistributedType(input.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")));
        var v_1 = IR.F.Math.Unary(UnaryOp.Cos, v_0);
        var v_2 = IR.F.Math.Unary(UnaryOp.Neg, v_0);
        var v_3 = IR.F.Math.Binary(BinaryOp.Add, v_1, v_2);
        var main = new Function("main", v_3, input);

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Unary>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallOp<Unary>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(2, pre_number);
        Assert.Equal(0, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestCircle2DiffModule()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var v_0 = IR.F.CPU.Boxing(input, new DistributedType(input.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")));
        var v_1 = IR.F.Math.Cos(IR.F.CPU.Boxing(v_0, input.CheckedType));
        var v_2 = IR.F.Math.Sin(v_0);
        var v_3 = IR.F.Math.Add(IR.F.CPU.Boxing(v_1, new DistributedType(input.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t"))), v_2);
        var v_4 = IR.F.Math.Neg(v_3);
        var main = new Function("main", v_4, new[] { input });

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Unary>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallOp<Unary>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(3, pre_number);
        Assert.Equal(1, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestCircle3SameModule()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var v_0 = IR.F.CPU.Boxing(input, new DistributedType(input.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")));
        var v_1 = IR.F.Math.Unary(UnaryOp.Abs, v_0);
        var v_2 = IR.F.Math.Unary(UnaryOp.Cos, v_0);
        var v_3 = IR.F.Math.Unary(UnaryOp.Neg, v_2);
        var v_4 = IR.F.Math.Binary(BinaryOp.Add, v_1, v_3);
        var main = new Function("main", v_4, new[] { input });

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Unary>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();
        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallOp<Unary>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(3, pre_number);
        Assert.Equal(0, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestCircle4SameModule()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var v_0 = IR.F.CPU.Boxing(input, new DistributedType(input.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")));
        var v_1 = IR.F.Math.Unary(UnaryOp.Abs, v_0);
        var v_2 = IR.F.Math.Unary(UnaryOp.Cos, v_1);
        var v_3 = IR.F.Math.Binary(BinaryOp.Add, v_0, v_2);
        var main = new Function("main", v_3, new[] { input });

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Unary>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();
        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallOp<Unary>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(2, pre_number);
        Assert.Equal(0, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestCircle5SameModule()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var v_0 = IR.F.CPU.Boxing(input, new DistributedType(input.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")));
        var v_1 = IR.F.CPU.Boxing(input, new DistributedType(input.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")));
        var v_2 = IR.F.Math.Binary(BinaryOp.Add, v_0, v_1);
        var main = new Function("main", v_2, new[] { input });

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Binary>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();
        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallOp<Unary>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(1, pre_number);
        Assert.Equal(0, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestTuple1SameModule()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var v_0 = IR.F.CPU.Boxing(input, new DistributedType(input.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")));
        var v_1 = IR.F.CPU.Boxing(input, new DistributedType(input.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")));
        var v_2 = new IR.Tuple(v_0, v_1);
        var main = new Function("main", v_2, new[] { input });
        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Boxing>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();
        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(2, pre_number);
        Assert.Equal(1, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestTuple2SameModule()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var v_0 = IR.F.CPU.Boxing(input, new DistributedType(input.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")));
        var v_1 = IR.F.Math.Unary(UnaryOp.Abs, v_0);
        var v_2 = IR.F.Math.Unary(UnaryOp.Abs, v_1);
        var v_3 = new IR.Tuple(v_0, v_2);
        var main = new Function("main", v_3, new[] { input });

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Unary>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();
        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(2, pre_number);
        Assert.Equal(1, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestConcat1SameModule()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var v_0 = IR.F.CPU.Boxing(input, new DistributedType(input.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")));
        var v_1 = IR.F.CPU.Boxing(input, new DistributedType(input.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")));
        var v_2 = new Call(new IR.Tensors.Concat(2), new IR.Tuple(v_0, v_1));
        var main = new Function("main", v_2, new[] { input });

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Boxing>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallOp<Boxing>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(2, pre_number);
        Assert.Equal(0, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestConcat2SameModule()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var v_0 = IR.F.CPU.Boxing(input, new DistributedType(input.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")));
        var v_1 = IR.F.Math.Unary(UnaryOp.Abs, v_0);
        var v_2 = IR.F.Math.Unary(UnaryOp.Cos, v_0);
        var v_3 = new Call(new IR.Tensors.Concat(2), new IR.Tuple(v_1, v_2));
        var main = new Function("main", v_3, new[] { input });

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Unary>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallOp<Unary>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(2, pre_number);
        Assert.Equal(0, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestConcat3SameModule()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var v_0 = new Call(new IR.Tensors.Concat(0), new IR.Tuple(input, input, input));
        var main = new Function("main", v_0, input);

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<IR.Tensors.Concat>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallOp<IR.Tensors.Concat>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(1, pre_number);
        Assert.Equal(1, post_number);
        Assert.Equal(pre_result, post_result);
    }

    [Fact]
    public async Task TestSplitSameModule()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 2, 32, 32 }));
        var v_0 = IR.F.Tensors.Split(input, 0, new[] { 1, 1 });
        var v_1 = IR.F.Tensors.GetItem(v_0, 0);
        var v_2 = IR.F.Tensors.GetItem(v_0, 1);
        var v_3 = new IR.Tuple(v_1, v_2);
        var main = new Function("main", v_3, input);

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(false);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<IR.Tensors.GetItem>();

        var input_tensor = Testing.Rand<float>(2, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<Nncase.Passes.CPUFunctionPartitionPass>();
        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallOp<IR.Tensors.GetItem>();
        var post_result = CompilerServices.Evaluate(((Function)module.Entry!).Body, feed_dict);

        Assert.Equal(2, pre_number);
        Assert.Equal(2, post_number);
        Assert.Equal(pre_result, post_result);
    }
}
