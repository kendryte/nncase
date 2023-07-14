// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using DryIoc;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Passes;
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
        pmgr.Add<EGraphExtractPass>();

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
        pmgr.Add<EGraphExtractPass>();

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
        prmg.Add<EGraphExtractPass>();

        await prmg.RunAsync(module);

        var post = (Function)module.Entry!;

        var post_result = CompilerServices.Evaluate(post.Body, feed_dict);

        // note 这里他其实强行分成了两个分支, fusion_1_2_3 和 fusion_2_fusion_1, 虽然结果一致但是不合理.
        Assert.True(Comparator.AllEqual(pre_result, post_result));
    }
}


internal sealed class SingleInputFusionMergeRule : IRewriteRule
{
    private readonly HashSet<string> _mergedCache = new();

    public IPattern Pattern { get; } =
      IsCall(
        "caller",
        IsFusion("caller_fusion", Callable.StackVMModuleKind, IsWildcard(), IsVArgs(IsVar())),
        IsCall(
          "callee",
          IsFusion("callee_fusion", Callable.StackVMModuleKind, IsWildcard(), IsVArgs(IsVar())),
          IsWildcard("callee_input")));

    public static Fusion MergeSingleInputFusion(string name, Call caller, Call callee, Fusion caller_fusion, Fusion callee_fusion, RunPassContext passOptions)
    {
        if (callee_fusion.Parameters.Length != 1 || caller_fusion.Parameters.Length != 1)
        {
            throw new NotSupportedException("Not Support Multi Inputs Fusion Merge");
        }

        // 1. replace the caller_fusion input_var with the callee_fusion body
        var new_fusion_body = new FusionMerger(new Dictionary<Var, Expr>(ReferenceEqualityComparer.Instance) { { caller_fusion.Parameters[0], callee_fusion.Body } }).Clone(caller_fusion.Body, default);
        new_fusion_body = CompilerServices.Rewrite(new_fusion_body, new[] { new Passes.Rules.Neutral.FoldTwoClamp() }, passOptions);
        // 2. fold the store load
        // new_fusion_body = CompilerServices.Rewrite(new_fusion_body, new[] { new Passes.Rules.K510.FoldStoreLoad() }, passOptions.IndentDir("MergeSingleInputFusion"));
        var new_fusion = new Fusion(name, Callable.StackVMModuleKind, new_fusion_body, callee_fusion.Parameters);

        return new_fusion;
    }

    public Expr? GetReplace(IMatchResult result, RunPassContext passOptions)
    {
        var caller = (Call)result["caller"];
        var callee = (Call)result["callee"];
        var callee_fusion = (Fusion)result["callee_fusion"];
        var caller_fusion = (Fusion)result["caller_fusion"];

        // note manual pruning
        if ((caller_fusion.Name.Split("_").Length +
            caller_fusion.Name.Split("_").Length) > (7 * 2))
        {
            return null;
        }

        // note each patter will generator the new expr. so need cache it.
        var name = $"{caller_fusion.Name}_{callee_fusion.Name}";
        if (_mergedCache.Contains(name))
        {
            return null;
        }

        // 1. merge new fusion
        var merged_fusion = MergeSingleInputFusion(name, caller, callee, caller_fusion, callee_fusion, passOptions);

        var new_call = new Call(merged_fusion, (Expr)result["callee_input"]);
        _mergedCache.Add(name);

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
        IsFusion("caller_fusion", Callable.StackVMModuleKind, IsWildcard(), IsVArgs(IsVar(), IsVar())),
        IsCall(
          "lhs_callee",
          IsFusion("lhs_callee_fusion", Callable.StackVMModuleKind, IsWildcard(), IsVArgs(IsVar())),
          _input),
        IsCall(
          "rhs_callee",
          IsFusion("rhs_callee_fusion", Callable.StackVMModuleKind, IsWildcard(), IsVArgs(IsVar())),
          _input));

    public static Fusion MergeTwoInputFusion(Call caller, Call lhs_callee, Call rhs_callee, Fusion caller_fusion, Fusion lhs_callee_fusion, Fusion rhs_callee_fusion, RunPassContext passOptions)
    {
        // 1. replace the caller_fusion input_var with the callee_fusion body
        var new_fusion_body = new FusionMerger(new Dictionary<Var, Expr>(ReferenceEqualityComparer.Instance) { { caller_fusion.Parameters[0], lhs_callee_fusion.Body }, { caller_fusion.Parameters[1], rhs_callee_fusion.Body }, }).Clone(caller_fusion.Body, default);

        // 2. fold the store load
        // new_fusion_body = CompilerServices.Rewrite(new_fusion_body, new[] { new Passes.Rules.K510.FoldStoreLoad() }, passOptions.IndentDir("MergeSingleInputFusion"));
        var new_fusion = new Fusion($"{caller_fusion.Name}_{lhs_callee_fusion.Name}_{rhs_callee_fusion.Name}", Callable.StackVMModuleKind, new_fusion_body, lhs_callee_fusion.Parameters);

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

sealed class SimpleClass
{
    CompileOptions Options { get; }

    public SimpleClass(CompileOptions options)
    {
        Options = options;
    }

    public string CW()
    {
        return Options.DumpDir;
    }
}

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestEGraphOnLineFusion : TestClassBase
{
    public UnitTestEGraphOnLineFusion()
    {
#if DEBUG
        CompileOptions.DumpFlags = DumpFlags.PassIR | DumpFlags.EGraphCost | DumpFlags.Rewrite;
#endif
    }

    [Fact]
    public void TestResolve()
    {
        var e = CompileSession.Resolve<ICostEvaluateProvider>();
        Assert.Equal("CostEvaluateProvider", e.GetType().Name);

        var e1 = CompileSession.Resolve<ICostEvaluateProvider>(new object[] { CostModelTest.SimulatorServer.LocalHost, (IRModule _) => string.Empty }, serviceKey: Evaluator.CostEvaluatorKinds.Online);
        Assert.Equal("OnlineCostEvaluateProvider", e1.GetType().Name);
    }

    [Fact]
    public async Task TestResNet18FusionOnlineCost()
    {
        var server = new CostModelTest.SimulatorServer(CostModelTest.SimulatorServer.LocalHost);

        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var model = new ResNet(typeof(BasicBlock), new[] { 2, 2, 2, 2 });
        var body = model.Forward(input);
        var main = new Function("main", body, input);

        var tv = new TestVisitor();
        tv.Visit(main);
        var pre_number = tv.CountCallOp<Conv2D>();

        var moduleCompile = (IRModule m) =>
        {
            var compiler = CompileSession.New<ICompiler>();
            compiler.ImportIRModule(m);
            var path = Path.GetTempFileName();
            using (var fs = File.OpenWrite(path))
            {
                compiler.Gencode(fs);
            }
            return path;
        };

        var module = new IRModule(main);
        var pmgr = CompileSession.CreatePassManager("pmgr");
        pmgr.AddWithName<EGraphRulesPass>("AutoMergeFusion").Configure(p =>
        {
            p.Add<SingleInputFusionMergeRule>();
        });
        pmgr.Add<EGraphExtractPass>().Configure(p =>
        {
            p.Extractor.CostEvaluateProvider = CompileSession.Resolve<ICostEvaluateProvider>(new object[] { CostModelTest.SimulatorServer.LocalHost, moduleCompile }, serviceKey: CostEvaluatorKinds.Online);
        });

        await pmgr.RunAsync(module);

        tv.Clear();
        tv.Visit(module.Entry!);
        var post_number = tv.CountCallOp<Conv2D>();
        Assert.Equal(pre_number, post_number);
    }

    [Fact]
    public async Task TestDataFlowFusionCycleFailedCase()
    {
        var server = new CostModelTest.SimulatorServer(CostModelTest.SimulatorServer.LocalHost);
        var moduleCompile = (IRModule m) =>
        {
            var compiler = CompileSession.New<ICompiler>();
            compiler.ImportIRModule(m);
            var path = Path.GetTempFileName();
            using (var fs = File.OpenWrite(path))
            {
                compiler.Gencode(fs);
            }
            return path;
        };

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
            p.Extractor.CostEvaluateProvider = CompileSession.Resolve<ICostEvaluateProvider>(new object[] { CostModelTest.SimulatorServer.LocalHost, moduleCompile }, serviceKey: CostEvaluatorKinds.Online);
        });

        await prmg.RunAsync(module);

        var post = (Function)module.Entry!;

        var post_result = CompilerServices.Evaluate(post.Body, feed_dict);

        // note 这里他其实强行分成了两个分支, fusion_1_2_3 和 fusion_2_fusion_1, 虽然结果一致但是不合理.
        Assert.True(Comparator.AllEqual(pre_result, post_result));
    }
}

internal sealed class FusionMerger : ExprCloner<Unit>
{
    private readonly Dictionary<Var, Expr> _varMap;

    public FusionMerger(Dictionary<Var, Expr> varMap)
    {
        _varMap = varMap;
    }

    protected override Expr VisitLeafVar(Var v, Unit context)
    {
        if (_varMap.TryGetValue(v, out var new_expr))
        {
            return Visit(new_expr, context);
        }

        return v;
    }
}
