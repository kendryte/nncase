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
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
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
    public async Task TestResNet18Fusion()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var model = new ResNet(typeof(BasicBlock), new[] { 2, 2, 2, 2 });
        var body = model.Forward(input);
        var main = new Function("main", body, input);

        var tv = new TestVisitor();
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Conv2D>();

        var module = new IRModule(main);
        var pmgr = CompileSession.CreatePassManager("pmgr");
        pmgr.Add<DataflowPass>().Configure(p =>
        {
            p.AddAnalysis<IExprUserAnalysisResult>();
            p.Add<DeterminedFusionMergeRule>();
        });
        pmgr.Add<EGraphRulesPass>().Configure(p =>
        {
            p.Add<GeneralFusionMergeRule>();
            p.Add<TupleFusionMergeRule>();
            p.Add<ConcatFusionMergeRule>();
        });
        pmgr.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await pmgr.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();
        Assert.Equal(1, post_number);
    }

    [Fact]
    public async Task TestResNet18FusionWithCycle()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var model = new ResNet(typeof(BasicBlock), new[] { 2, 2, 2, 2 });
        var body = model.Forward(input);
        var main = new Function("main", body, input);

        var tv = new TestVisitor(true);
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Conv2D>();

        var module = new IRModule(main);
        var pmgr = CompileSession.CreatePassManager("pmgr");
        pmgr.Add<DataflowPass>().Configure(p =>
        {
            p.AddAnalysis<IExprUserAnalysisResult>();
            p.Add<DeterminedFusionMergeRule>();
        });
        pmgr.AddWithName<EGraphRulesPass>("AutoMergeFusion").Configure(p =>
        {
            p.Add<GeneralFusionMergeRule>();
            p.Add<TupleFusionMergeRule>();
            p.Add<ConcatFusionMergeRule>();
        });
        pmgr.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await pmgr.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();

        // note when the load store cost > recompute, so the post number will > pre number!.
        Assert.Equal(1, post_number);
    }

    /// <summary>
    /// cycle type 1.
    ///  in dataflow pass, will merge fusion1_2_3, then it will run duplicate fusion_1,fusion_2.
    ///  but in egraph pass, we have no need using user analysis.
    ///             x = fusion1(input)
    ///               |
    ///             z = fusion2(y)
    ///            /    \
    ///         /         \
    ///        |      m = fusion2(z)
    ///         \        /
    ///          \     /
    ///     fusion3(z,m).
    /// </summary>
    /// <returns>A <see cref="Task"/> representing the asynchronous unit test.</returns>
    [Fact]
    public async Task TestDataFlowFusionCycleFailedCase()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 224, 224, 3 }));
        Function main;
        {
            var fusion_1_input = new Var("fusion_1_input", new TensorType(DataTypes.Float32, new int[] { 1, 224, 224, 3 }));
            var fusion_1 = new Fusion("fusion_1", Callable.StackVMModuleKind, IR.F.Tensors.NHWCToNCHW(fusion_1_input), new[] { fusion_1_input });
            var v_0 = new Call(fusion_1, input); // 1,3,224,224

            var fusion_2_input = new Var("fusion_2_input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
            var fusion_2 = new Fusion(
                "fusion_2",
                Callable.StackVMModuleKind,
                IR.F.NN.ReduceWindow2D(
                    ReduceOp.Max,
                    fusion_2_input,
                    0.0f,
                    new[] { 3, 3 },
                    new[] { 2, 2 },
                    new[,]
                    {
                        { 1, 1 },
                        { 1, 1 },
                    },
                    new[] { 1, 1 },
                    false,
                    false),
                new[] { fusion_2_input });
            var v_1 = new Call(fusion_2, v_0); // 1,3,112,112

            var fusion_3_input = new Var("fusion_3_input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 112, 112 }));
            var fusion_3 = new Fusion(
                "fusion_3",
                Callable.StackVMModuleKind,
                IR.F.NN.ReduceWindow2D(
                    ReduceOp.Mean,
                    fusion_3_input,
                    0.0f,
                    new[] { 3, 3 },
                    new[] { 1, 1 },
                    new[,]
                    {
                        { 1, 1 },
                        { 1, 1 },
                    },
                    new[] { 1, 1 },
                    false,
                    false),
                new[] { fusion_3_input });
            var v_2 = new Call(fusion_3, v_1); // 1,3,112,112

            var fusion_4_input = new Var[] { new("fusion_4_input_0", new TensorType(DataTypes.Float32, new int[] { 1, 3, 112, 112 })), new("fusion_4_input_1", new TensorType(DataTypes.Float32, new int[] { 1, 3, 112, 112 })) };
            var fusion_4 = new Fusion("fusion_4", Callable.StackVMModuleKind, IR.F.Math.Add(fusion_4_input[0], fusion_4_input[1]), fusion_4_input);
            var v_3 = new Call(fusion_4, new[] { v_1, v_2 }); // 1,3,112,112
            main = new Function("main", v_3, input);
        }

        Assert.True(CompilerServices.InferenceType(main));

        var input_tensor = Testing.Rand<float>(1, 224, 224, 3);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
          { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var test_visitor = new TestVisitor(true);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<EGraphRulesPass>().Configure(p =>
        {
            // p.Add<SingleInputFusionMergeRule>();
        });
        prmg.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await prmg.RunAsync(module);

        var post = (Function)module.Entry!;

        var post_result = CompilerServices.Evaluate(post.Body, feed_dict);

        // note 这里他其实强行分成了两个分支, fusion_1_2_3 和 fusion_2_fusion_1, 虽然结果一致但是不合理.
        Assert.True(Comparator.AllEqual(pre_result, post_result));
    }

    [Fact]
    public async Task TestLineSameModule()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var main = new Function("main", IR.F.Math.Unary(UnaryOp.Abs, IR.F.Math.Unary(UnaryOp.Sin, input)), input);

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(true);
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

        Assert.Equal(2, pre_number);
        Assert.Equal(1, post_number);
    }

    [Fact]
    public async Task TestLineDiffModule()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var main = new Function("main", IR.F.Math.Abs(IR.F.CPU.Boxing(input, new DistributedType(input.CheckedTensorType, new[] { SBP.B }, new(new[] { 1 }, "t")))), input);

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(true);
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

        Assert.Equal(1, pre_number);
        Assert.Equal(0, post_number);
    }

    [Fact]
    public async Task TestYSameModule()
    {
        // step 1. import
        var input1 = new Var("input1", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var input2 = new Var("input2", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        Function main;
        {
            var fusion_2_input = new Var("fusion_2_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_2 = new Fusion("fusion_2", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Cos, fusion_2_input), new[] { fusion_2_input });
            var v_1 = new Call(fusion_2, input1);

            var fusion_3_input = new Var("fusion_3_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_3 = new Fusion("fusion_3", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Neg, fusion_3_input), new[] { fusion_3_input });
            var v_2 = new Call(fusion_3, input2);

            var fusion_4_input_0 = new Var("fusion_4_input_0", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_4_input_1 = new Var("fusion_4_input_1", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_4 = new Fusion("fusion_4", Callable.StackVMModuleKind, IR.F.Math.Binary(BinaryOp.Add, fusion_4_input_0, fusion_4_input_1), new[] { fusion_4_input_0, fusion_4_input_1 });
            var v_3 = new Call(fusion_4, v_1, v_2);

            main = new Function("main", v_3, input1, input2);
        }

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(true);
        tv.Visit(main);
        var pre_number = tv.CountCallFusion<Fusion>();

        var input_tensor1 = Testing.Rand<float>(1, 32, 32);
        var input_tensor2 = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input1, Value.FromTensor(input_tensor1) },
            { input2, Value.FromTensor(input_tensor2) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var test_visitor = new TestVisitor(true);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<EGraphRulesPass>().Configure(p =>
        {
            p.Add<GeneralFusionMergeRule>();
        });
        prmg.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();

        Assert.Equal(3, pre_number);
        Assert.Equal(1, post_number);
    }

    [Fact]
    public async Task TestHandInHandSameModule()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        Function main;
        {
            var fusion_1_input = new Var("fusion_1_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_1 = new Fusion("fusion_1", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Abs, fusion_1_input), new[] { fusion_1_input });
            var v_0 = new Call(fusion_1, input);

            var fusion_2_input_0 = new Var("fusion_2_input_0", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_2_input_1 = new Var("fusion_2_input_1", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_2 = new Fusion("fusion_2", Callable.StackVMModuleKind, IR.F.Math.Binary(BinaryOp.Add, IR.F.Math.Unary(UnaryOp.Cos, fusion_2_input_0), IR.F.Math.Unary(UnaryOp.Sin, fusion_2_input_1)), new[] { fusion_2_input_0, fusion_2_input_1 });
            var v_1 = new Call(fusion_2, v_0, v_0);

            main = new Function("main", v_1, input);
        }

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(true);
        tv.Visit(main);
        var pre_number = tv.CountCallFusion<Fusion>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var test_visitor = new TestVisitor(true);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<EGraphRulesPass>().Configure(p =>
        {
            p.Add<GeneralFusionMergeRule>();
        });
        prmg.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();

        Assert.Equal(2, pre_number);
        Assert.Equal(1, post_number);
    }

    [Fact]
    public async Task TestCircle1SameModule()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        Function main;
        {
            var fusion_1_input = new Var("fusion_1_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_1 = new Fusion("fusion_1", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Abs, fusion_1_input), new[] { fusion_1_input });
            var v_0 = new Call(fusion_1, input);

            var fusion_2_input = new Var("fusion_2_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_2 = new Fusion("fusion_2", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Cos, fusion_2_input), new[] { fusion_2_input });
            var v_1 = new Call(fusion_2, v_0);

            var fusion_3_input_0 = new Var("fusion_3_input_0", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_3_input_1 = new Var("fusion_3_input_1", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_3 = new Fusion("fusion_3", Callable.StackVMModuleKind, IR.F.Math.Binary(BinaryOp.Add, fusion_3_input_0, fusion_3_input_1), new[] { fusion_3_input_0, fusion_3_input_1 });
            var v_2 = new Call(fusion_3, v_0, v_1);

            main = new Function("main", v_2, input);
        }

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(true);
        tv.Visit(main);
        var pre_number = tv.CountCallFusion<Fusion>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var test_visitor = new TestVisitor(true);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<EGraphRulesPass>().Configure(p =>
        {
            p.Add<GeneralFusionMergeRule>();
        });
        prmg.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();

        Assert.Equal(3, pre_number);
        Assert.Equal(1, post_number);
    }

    [Fact]
    public async Task TestCircle2SameModule()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        Function main;
        {
            var fusion_1_input = new Var("fusion_1_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_1 = new Fusion("fusion_1", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Abs, fusion_1_input), new[] { fusion_1_input });
            var v_0 = new Call(fusion_1, input);

            var fusion_2_input = new Var("fusion_2_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_2 = new Fusion("fusion_2", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Cos, fusion_2_input), new[] { fusion_2_input });
            var v_1 = new Call(fusion_2, v_0);

            var fusion_3_input = new Var("fusion_3_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_3 = new Fusion("fusion_3", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Neg, fusion_3_input), new[] { fusion_3_input });
            var v_2 = new Call(fusion_3, v_0);

            var fusion_4_input_0 = new Var("fusion_4_input_0", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_4_input_1 = new Var("fusion_4_input_1", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_4 = new Fusion("fusion_4", Callable.StackVMModuleKind, IR.F.Math.Binary(BinaryOp.Add, fusion_4_input_0, fusion_4_input_1), new[] { fusion_4_input_0, fusion_4_input_1 });
            var v_3 = new Call(fusion_4, v_1, v_2);

            main = new Function("main", v_3, input);
        }

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(true);
        tv.Visit(main);
        var pre_number = tv.CountCallFusion<Fusion>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var test_visitor = new TestVisitor(true);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<EGraphRulesPass>().Configure(p =>
        {
            p.Add<GeneralFusionMergeRule>();
        });
        prmg.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();

        Assert.Equal(4, pre_number);
        Assert.Equal(1, post_number);
    }

    [Fact]
    public async Task TestCircle2DiffModule()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        var v_0 = IR.F.Math.Abs(input);
        var v_1 = IR.F.Math.Cos(v_0);
        var v_2 = IR.F.CPU.Boxing(v_0, v_0.CheckedTensorType);
        var v_3 = IR.F.Math.Add(v_1, v_2);
        var main = new Function("main", v_3, new[] { input });

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(true);
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

        Assert.Equal(2, pre_number);
        Assert.Equal(0, post_number);
    }

    [Fact]
    public async Task TestCircle3SameModule()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        Function main;
        {
            var fusion_1_input = new Var("fusion_1_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_1 = new Fusion("fusion_1", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Abs, fusion_1_input), new[] { fusion_1_input });
            var v_0 = new Call(fusion_1, input);

            var fusion_2_input = new Var("fusion_2_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_2 = new Fusion("fusion_2", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Cos, fusion_2_input), new[] { fusion_2_input });
            var v_1 = new Call(fusion_2, v_0);

            var fusion_3_input = new Var("fusion_3_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_3 = new Fusion("fusion_3", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Neg, fusion_3_input), new[] { fusion_3_input });
            var v_2 = new Call(fusion_3, v_0);

            var fusion_4_input = new Var("fusion_4_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_4 = new Fusion("fusion_4", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Neg, fusion_4_input), new[] { fusion_4_input });
            var v_3 = new Call(fusion_4, v_2);

            var fusion_5_input_0 = new Var("fusion_5_input_0", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_5_input_1 = new Var("fusion_5_input_1", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_5 = new Fusion("fusion_5", Callable.StackVMModuleKind, IR.F.Math.Binary(BinaryOp.Add, fusion_5_input_0, fusion_5_input_1), new[] { fusion_5_input_0, fusion_5_input_1 });
            var v_4 = new Call(fusion_5, v_1, v_3);

            main = new Function("main", v_4, input);
        }

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(true);
        tv.Visit(main);
        var pre_number = tv.CountCallFusion<Fusion>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var test_visitor = new TestVisitor(true);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<EGraphRulesPass>().Configure(p =>
        {
            p.Add<GeneralFusionMergeRule>();
        });
        prmg.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();

        Assert.Equal(5, pre_number);
        Assert.Equal(1, post_number);
    }

    [Fact]
    public async Task TestCircle4SameModule()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        Function main;
        {
            var fusion_1_input = new Var("fusion_1_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_1 = new Fusion("fusion_1", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Abs, fusion_1_input), new[] { fusion_1_input });
            var v_0 = new Call(fusion_1, input);

            var fusion_2_input = new Var("fusion_2_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_2 = new Fusion("fusion_2", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Cos, fusion_2_input), new[] { fusion_2_input });
            var v_1 = new Call(fusion_2, v_0);

            var fusion_3_input = new Var("fusion_3_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_3 = new Fusion("fusion_3", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Sin, fusion_3_input), new[] { fusion_3_input });
            var v_2 = new Call(fusion_3, v_1);

            var fusion_4_input_0 = new Var("fusion_4_input_0", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_4_input_1 = new Var("fusion_4_input_1", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_4 = new Fusion("fusion_4", Callable.StackVMModuleKind, IR.F.Math.Binary(BinaryOp.Add, fusion_4_input_0, fusion_4_input_1), new[] { fusion_4_input_0, fusion_4_input_1 });
            var v_3 = new Call(fusion_4, v_0, v_2);

            main = new Function("main", v_3, input);
        }

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(true);
        tv.Visit(main);
        var pre_number = tv.CountCallFusion<Fusion>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var test_visitor = new TestVisitor(true);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<EGraphRulesPass>().Configure(p =>
        {
            p.Add<GeneralFusionMergeRule>();
        });
        prmg.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();

        Assert.Equal(4, pre_number);
        Assert.Equal(1, post_number);
    }

    [Fact]
    public async Task TestCircle5SameModule()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        Function main;
        {
            var fusion_1_input = new Var("fusion_1_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_1 = new Fusion("fusion_1", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Abs, fusion_1_input), new[] { fusion_1_input });
            var v_0 = new Call(fusion_1, input);

            var fusion_2_input = new Var("fusion_2_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_2 = new Fusion("fusion_2", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Cos, fusion_2_input), new[] { fusion_2_input });
            var v_1 = new Call(fusion_2, input);

            var fusion_3_input_0 = new Var("fusion_3_input_0", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_3_input_1 = new Var("fusion_3_input_1", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_3 = new Fusion("fusion_3", Callable.StackVMModuleKind, IR.F.Math.Binary(BinaryOp.Add, fusion_3_input_0, fusion_3_input_1), new[] { fusion_3_input_0, fusion_3_input_1 });
            var v_2 = new Call(fusion_3, v_0, v_1);

            main = new Function("main", v_2, input);
        }

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(true);
        tv.Visit(main);
        var pre_number = tv.CountCallFusion<Fusion>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var test_visitor = new TestVisitor(true);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<EGraphRulesPass>().Configure(p =>
        {
            p.Add<GeneralFusionMergeRule>();
        });
        prmg.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();

        Assert.Equal(3, pre_number);
        Assert.Equal(1, post_number);
    }

    [Fact]
    public async Task TestTuple1SameModule()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        Function main;
        {
            var fusion_1_input = new Var("fusion_1_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_1 = new Fusion("fusion_1", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Abs, fusion_1_input), new[] { fusion_1_input });
            var v_0 = new Call(fusion_1, input);

            var fusion_2_input = new Var("fusion_2_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_2 = new Fusion("fusion_2", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Cos, fusion_2_input), new[] { fusion_2_input });
            var v_1 = new Call(fusion_2, input);

            var v_2 = new IR.Tuple(v_0, v_1);
            main = new Function("main", v_2, input);
        }

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(true);
        tv.Visit(main);
        var pre_number = tv.CountCallFusion<Fusion>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var test_visitor = new TestVisitor(true);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<EGraphRulesPass>().Configure(p =>
        {
            p.Add<GeneralFusionMergeRule>();
            p.Add<TupleFusionMergeRule>();
        });
        prmg.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();

        Assert.Equal(2, pre_number);
        Assert.Equal(1, post_number);
    }

    [Fact]
    public async Task TestTuple2SameModule()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        Function main;
        {
            var fusion_1_input = new Var("fusion_1_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_1 = new Fusion("fusion_1", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Abs, fusion_1_input), new[] { fusion_1_input });
            var v_0 = new Call(fusion_1, input);

            var fusion_2_input = new Var("fusion_2_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_2 = new Fusion("fusion_2", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Cos, fusion_2_input), new[] { fusion_2_input });
            var v_1 = new Call(fusion_2, v_0);

            var fusion_3_input = new Var("fusion_3_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_3 = new Fusion("fusion_3", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Abs, fusion_3_input), new[] { fusion_3_input });
            var v_2 = new Call(fusion_3, v_0);

            var v_3 = new IR.Tuple(v_1, v_2);
            main = new Function("main", v_3, input);
        }

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(true);
        tv.Visit(main);
        var pre_number = tv.CountCallFusion<Fusion>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var test_visitor = new TestVisitor(true);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<EGraphRulesPass>().Configure(p =>
        {
            p.Add<GeneralFusionMergeRule>();
            p.Add<TupleFusionMergeRule>();
        });
        prmg.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();

        Assert.Equal(3, pre_number);
        Assert.Equal(1, post_number);
    }

    [Fact]
    public async Task TestConcat1SameModule()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        Function main;
        {
            var fusion_1_input = new Var("fusion_1_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_1 = new Fusion("fusion_1", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Abs, fusion_1_input), new[] { fusion_1_input });
            var v_0 = new Call(fusion_1, input);

            var fusion_2_input = new Var("fusion_2_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_2 = new Fusion("fusion_2", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Cos, fusion_2_input), new[] { fusion_2_input });
            var v_1 = new Call(fusion_2, input);

            var v_2 = new Call(new IR.Tensors.Concat(2), new IR.Tuple(v_0, v_1));
            main = new Function("main", v_2, input);
        }

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(true);
        tv.Visit(main);
        var pre_number = tv.CountCallFusion<Fusion>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var test_visitor = new TestVisitor(true);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<EGraphRulesPass>().Configure(p =>
        {
            p.Add<GeneralFusionMergeRule>();
            p.Add<TupleFusionMergeRule>();
            p.Add<ConcatFusionMergeRule>();
        });
        prmg.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();

        Assert.Equal(2, pre_number);
        Assert.Equal(1, post_number);
    }

    [Fact]
    public async Task TestConcat2SameModule()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        Function main;
        {
            var fusion_1_input = new Var("fusion_1_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_1 = new Fusion("fusion_1", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Abs, fusion_1_input), new[] { fusion_1_input });
            var v_0 = new Call(fusion_1, input);

            var fusion_2_input = new Var("fusion_2_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_2 = new Fusion("fusion_2", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Cos, fusion_2_input), new[] { fusion_2_input });
            var v_1 = new Call(fusion_2, v_0);

            var fusion_3_input = new Var("fusion_3_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_3 = new Fusion("fusion_3", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Abs, fusion_3_input), new[] { fusion_3_input });
            var v_2 = new Call(fusion_3, v_0);

            var v_3 = new Call(new IR.Tensors.Concat(2), new IR.Tuple(v_1, v_2));
            main = new Function("main", v_3, input);
        }

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(true);
        tv.Visit(main);
        var pre_number = tv.CountCallFusion<Fusion>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var test_visitor = new TestVisitor(true);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<EGraphRulesPass>().Configure(p =>
        {
            p.Add<GeneralFusionMergeRule>();
            p.Add<TupleFusionMergeRule>();
            p.Add<ConcatFusionMergeRule>();
        });
        prmg.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();

        Assert.Equal(3, pre_number);
        Assert.Equal(1, post_number);
    }

    [Fact]
    public async Task TestConcat3SameModule()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        Function main;
        {
            var v_0 = new Call(new IR.Tensors.Concat(0), new IR.Tuple(input, input, input));
            main = new Function("main", v_0, input);
        }

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(true);
        tv.Visit(main);
        var pre_number = tv.CountCallFusion<Fusion>();

        var input_tensor = Testing.Rand<float>(1, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var test_visitor = new TestVisitor(true);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<DataflowPass>().Configure(p =>
        {
            p.Add<CPUSingleFusion>(CPUTarget.Kind);
            p.Add<CPUOutputBoxingFusion>(CPUTarget.Kind);
        });
        prmg.Add<EGraphRulesPass>().Configure(p =>
        {
            p.Add<GeneralFusionMergeRule>();
            p.Add<TupleFusionMergeRule>();
            p.Add<ConcatFusionMergeRule>();
        });
        prmg.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();

        Assert.Equal(0, pre_number);
        Assert.Equal(1, post_number);
    }

    [Fact]
    public async Task TestSplitSameModule()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 2, 32, 32 }));
        Function main;
        {
            var v_0 = IR.F.Tensors.Split(input, 0, new[] { 1, 1 });

            var fusion_2_input = new Var("fusion_2_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_2 = new Fusion("fusion_2", CPUTarget.Kind, IR.F.Math.Unary(UnaryOp.Cos, fusion_2_input), new[] { fusion_2_input });
            var v_1 = new Call(fusion_2, IR.F.Tensors.GetItem(v_0, 0));

            var fusion_3_input = new Var("fusion_3_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_3 = new Fusion("fusion_3", CPUTarget.Kind, IR.F.Math.Unary(UnaryOp.Abs, fusion_3_input), new[] { fusion_3_input });
            var v_2 = new Call(fusion_3, IR.F.Tensors.GetItem(v_0, 1));

            var v_3 = new IR.Tuple(v_1, v_2);
            main = new Function("main", v_3, input);
        }

        Assert.True(CompilerServices.InferenceType(main));

        var tv = new TestVisitor(true);
        tv.Visit(main);
        var pre_number = tv.CountCallFusion<Fusion>() + tv.CountCallOp<IR.Tensors.Split>();

        var input_tensor = Testing.Rand<float>(2, 32, 32);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var test_visitor = new TestVisitor(true);
        var module = new IRModule(main);

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.Add<DataflowPass>().Configure(p =>
        {
            p.Add<CPUSingleFusion>(CPUTarget.Kind);
            p.Add<CPUOutputBoxingFusion>(CPUTarget.Kind);
        });
        DumpScope.Current.DumpIR(main, "pre");
        prmg.Add<EGraphRulesPass>().Configure(p =>
        {
            p.Add<GeneralFusionMergeRule>();
            p.Add<TupleFusionMergeRule>();
            p.Add<ConcatFusionMergeRule>();
        });

        DumpScope.Current.DumpIR(main, "post");
        prmg.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await prmg.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallFusion<Fusion>();

        Assert.Equal(3, pre_number);
        Assert.Equal(1, post_number);
    }
}
