// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Passes;
using Nncase.Passes.Rules.CPU;
using Nncase.PatternMatch;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.ReWrite.FusionTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestEGraphFusion : TestClassBase
{
    public UnitTestEGraphFusion()
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
        pmgr.AddWithName<EGraphRulesPass>("AutoMergeFusion").Configure(p =>
        {
            p.Add<SingleInputFusionMergeRule>();
        });
        pmgr.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await pmgr.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallOp<Conv2D>();
        Assert.Equal(pre_number, post_number);
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
        pmgr.AddWithName<EGraphRulesPass>("AutoMergeFusion").Configure(p =>
        {
            p.Add<SingleInputFusionMergeRule>();
            p.Add<TwoInputFusionMergeRule>();
        });
        pmgr.Add<EGraphExtractPass>().Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<FusionCostEvaluator>();
        });

        await pmgr.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallOp<Conv2D>();

        // note when the load store cost > recompute, so the post number will > pre number!.
        // Assert.Equal(pre_number, post_number);
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
            p.Add<SingleInputFusionMergeRule>();
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
            var v_2 = new Call(fusion_3, v_1);

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
    public async Task TestLineDiffModule()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
        Function main;
        {
            var fusion_1_input = new Var("fusion_1_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_1 = new Fusion("fusion_1", Callable.StackVMModuleKind, IR.F.Math.Unary(UnaryOp.Abs, fusion_1_input), new[] { fusion_1_input });
            var v_0 = new Call(fusion_1, input);

            var fusion_2_input = new Var("fusion_2_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_2 = new Fusion("fusion_2", "cpu", IR.F.Math.Unary(UnaryOp.Cos, fusion_2_input), new[] { fusion_2_input });
            var v_1 = new Call(fusion_2, v_0);

            var fusion_3_input = new Var("fusion_3_input", new TensorType(DataTypes.Float32, new int[] { 1, 32, 32 }));
            var fusion_3 = new Fusion("fusion_3", "cpu", IR.F.Math.Unary(UnaryOp.Neg, fusion_3_input), new[] { fusion_3_input });
            var v_2 = new Call(fusion_3, v_1);

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
        Assert.Equal(2, post_number);
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
}

internal sealed class SingleInputFusionMergeRule : IRewriteRule
{
    private readonly Dictionary<int, Call> _mergedCache = new();

    public IPattern Pattern { get; } =
      IsCall(
        "caller",
        IsFusion("caller_fusion", _ => true, IsWildcard(), IsVArgs(IsVar())),
        IsCall(
          "callee",
          IsFusion("callee_fusion", _ => true, IsWildcard(), IsVArgs(IsVar())),
          IsWildcard("callee_input")));

    public static Fusion MergeSingleInputFusion(Call caller, Call callee, Fusion caller_fusion, Fusion callee_fusion, RunPassContext passOptions)
    {
        if (callee_fusion.Parameters.Length != 1 || caller_fusion.Parameters.Length != 1)
        {
            throw new NotSupportedException("Not Support Multi Inputs Fusion Merge");
        }

        // 1. replace the caller_fusion input_var with the callee_fusion body
        var new_fusion_body = new FusionMerger(new Dictionary<Var, Expr>(ReferenceEqualityComparer.Instance) { { caller_fusion.Parameters[0], callee_fusion.Body } }).Clone(caller_fusion.Body, default);

        // 2. fold the store load
        // new_fusion_body = CompilerServices.Rewrite(new_fusion_body, new[] { new Passes.Rules.K510.FoldStoreLoad() }, passOptions.IndentDir("MergeSingleInputFusion"));
        var new_fusion = new Fusion($"{caller_fusion.Name}_{callee_fusion.Name}", caller_fusion.ModuleKind, new_fusion_body, callee_fusion.Parameters);

        return new_fusion;
    }

    public Expr? GetReplace(IMatchResult result, RunPassContext passOptions)
    {
        var caller = (Call)result["caller"];
        var callee = (Call)result["callee"];
        var callee_fusion = (Fusion)result["callee_fusion"];
        var caller_fusion = (Fusion)result["caller_fusion"];

        if (caller_fusion.ModuleKind != callee_fusion.ModuleKind)
        {
            return null;
        }

        // note manual pruning
        if ((caller_fusion.Name.Split("_").Length +
            caller_fusion.Name.Split("_").Length) > 7)
        {
            return caller;
        }

        // note each patter will generator the new expr. so need cache it.
        var hashcode = HashCode.Combine(ReferenceEqualityComparer.Instance.GetHashCode(caller), ReferenceEqualityComparer.Instance.GetHashCode(callee));
        if (!_mergedCache.TryGetValue(hashcode, out var new_call))
        {
            // 1. merge new fusion
            var merged_fusion = MergeSingleInputFusion(caller, callee, caller_fusion, callee_fusion, passOptions);

            // if (true)
            // {
            new_call = new Call(merged_fusion, (Expr)result["callee_input"]);
            _mergedCache.Add(hashcode, new_call);
        }
        else
        {
            // System.Console.WriteLine("Re Add Merged Fusion Call");
        }

        return new_call;
    }
}

/// <summary>
/// fusion_3(fusion_1(x), fusion_2(x)) => fusion_4(x).
/// </summary>
internal sealed class TwoInputFusionMergeRule : IRewriteRule
{
    private static readonly Pattern _input = IsWildcard("input");

    private readonly Dictionary<int, Call> _mergedCache = new();

    public IPattern Pattern { get; } =
      IsCall(
        "caller",
        IsFusion("caller_fusion", _ => true, IsWildcard(), IsVArgs(IsVar(), IsVar())),
        IsCall(
          "lhs_callee",
          IsFusion("lhs_callee_fusion", _ => true, IsWildcard(), IsVArgs(IsVar())),
          _input),
        IsCall(
          "rhs_callee",
          IsFusion("rhs_callee_fusion", _ => true, IsWildcard(), IsVArgs(IsVar())),
          _input));

    public static Fusion MergeTwoInputFusion(Call caller, Call lhs_callee, Call rhs_callee, Fusion caller_fusion, Fusion lhs_callee_fusion, Fusion rhs_callee_fusion, RunPassContext passOptions)
    {
        // 1. replace the caller_fusion input_var with the callee_fusion body
        var new_fusion_body = new FusionMerger(new Dictionary<Var, Expr>(ReferenceEqualityComparer.Instance) { { caller_fusion.Parameters[0], lhs_callee_fusion.Body }, { caller_fusion.Parameters[1], rhs_callee_fusion.Body }, }).Clone(caller_fusion.Body, default);

        // 2. fold the store load
        // new_fusion_body = CompilerServices.Rewrite(new_fusion_body, new[] { new Passes.Rules.K510.FoldStoreLoad() }, passOptions.IndentDir("MergeSingleInputFusion"));
        var new_fusion = new Fusion($"{caller_fusion.Name}_{lhs_callee_fusion.Name}_{rhs_callee_fusion.Name}", caller_fusion.ModuleKind, new_fusion_body, lhs_callee_fusion.Parameters);

        return new_fusion;
    }

    public Expr? GetReplace(IMatchResult result, RunPassContext options)
    {
        if (ReferenceEquals((Call)result["lhs_callee"], (Call)result["rhs_callee"]))
        {
            return null;
        }

        return GetReplace(
          (Call)result["caller"], (Call)result["lhs_callee"], (Call)result["rhs_callee"], (Fusion)result["caller_fusion"], (Fusion)result["lhs_callee_fusion"], (Fusion)result["rhs_callee_fusion"], (Expr)result["input"], options);
    }

    private Expr? GetReplace(Call caller, Call lhs_callee, Call rhs_callee, Fusion caller_fusion, Fusion lhs_callee_fusion, Fusion rhs_callee_fusion, Expr input, RunPassContext passOptions)
    {
        if (caller_fusion.ModuleKind != lhs_callee_fusion.ModuleKind || caller_fusion.ModuleKind != rhs_callee_fusion.ModuleKind)
        {
            return null;
        }

        // note manual pruning
        if ((caller_fusion.Name.Split("_").Length +
          lhs_callee_fusion.Name.Split("_").Length +
          rhs_callee_fusion.Name.Split("_").Length) > 7)
        {
            return caller;
        }

        // note each patter will generator the new expr. so need cache it.
        var hashcode = HashCode.Combine(
            ReferenceEqualityComparer.Instance.GetHashCode(caller_fusion),
            ReferenceEqualityComparer.Instance.GetHashCode(lhs_callee_fusion),
            ReferenceEqualityComparer.Instance.GetHashCode(rhs_callee_fusion));
        if (!_mergedCache.TryGetValue(hashcode, out var new_call))
        {
            // 1. merge new fusion
            var merged_fusion = MergeTwoInputFusion(caller, lhs_callee, rhs_callee, caller_fusion, lhs_callee_fusion, rhs_callee_fusion, passOptions);
            new_call = new Call(merged_fusion, input);
            _mergedCache.Add(hashcode, new_call);
        }
        else
        {
            // System.Console.WriteLine("Re Add Merged Two Fusion Call");
        }

        return new_call;
    }
}
