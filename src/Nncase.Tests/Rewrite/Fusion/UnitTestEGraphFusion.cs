// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
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
    [Fact]
    public async Task TestResNet18Fusion()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var model = new ResNet(typeof(BasicBlock), new[] { 2, 2, 2, 2 });
        var body = model.Forward(input);
        var main = new Function("main", body, input);

        Assert.True(CompilerServices.InferenceType(main));

        var pass = new EGraphRulesPass { Name = "AutoMergeFusion" };
        pass.Add<SingleInputFusionMergeRule>();

        var graph = new EGraph(main);
        await pass.RunAsync(graph, new());
        graph.Extract(graph.Root!, new FusionCostEvaluator());
    }

    [Fact]
    public async Task TestResNet18FusionWithCycle()
    {
        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var model = new ResNet(typeof(BasicBlock), new[] { 2, 2, 2, 2 });
        var body = model.Forward(input);
        var main = new Function("main", body, input);

        Assert.True(CompilerServices.InferenceType(main));

        var pass = new EGraphRulesPass { Name = "AutoMergeFusion" };
        pass.Add<SingleInputFusionMergeRule>();
        pass.Add<TwoInputFusionMergeRule>();

        var graph = new EGraph(main);
        await pass.RunAsync(graph, new());
        graph.Extract(graph.Root!, new FusionCostEvaluator());
    }

    /// <summary>
    /// cycle type 1:  这里会存在一个bug, 如果是dataflow的话, merge single input 就会把 x y 合并在一起. 需要知道use关系才行.
    ///             x = fusion1(input)
    ///            /    \
    ///         /         \
    ///        |      y = fusion2(x)
    ///         \        /
    ///          \     /
    ///     fusion3(x,y).
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

        var pass = new DataflowPass { Name = "AutoMergeFusion" };
        pass.Add<SingleInputFusionMergeRule>();
        var post = (Function)await pass.RunAsync(main, new());

        var pass2 = new EGraphRulesPass { Name = "EGraphAutoMergeFusion" };
        pass2.Add<SingleInputFusionMergeRule>();

        var graph = new EGraph(main);
        await pass2.RunAsync(graph, new());
        var post2 = (Function)graph.Extract(graph.Root!, new FusionCostEvaluator());

        var input_tensor = Testing.Rand<float>(1, 224, 224, 3);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
          { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var post_result = CompilerServices.Evaluate(post.Body, feed_dict);
        var post2_result = CompilerServices.Evaluate(post2.Body, feed_dict);

        // note 这里他其实强行分成了两个分支, fusion_1_2_3 和 fusion_2_fusion_1, 虽然结果一致但是不合理.
        Assert.True(Comparator.AllEqual(pre_result, post_result));
        Assert.True(Comparator.AllEqual(pre_result, post2_result));
    }
}

internal sealed class FusionCostEvaluator : Evaluator.IBaseFuncCostEvaluator
{
    public Cost VisitLeaf(BaseFunction target)
    {
        if (target is Fusion fusion && fusion.ModuleKind == Callable.StackVMModuleKind)
        {
            return new FusionGraphCostVisitor().Visit(fusion);
        }
        else
        {
            throw new NotSupportedException();
        }
    }

    private sealed class GraphOpCostEvaluateContext : Evaluator.ICostEvaluateContext
    {
        private readonly IRType? _returnType;
        private readonly IRType?[] _argumentTypes;
        private readonly Expr[] _arguments;

        public GraphOpCostEvaluateContext(IRType? returnType, IRType?[] argumentTypes, ReadOnlySpan<Expr> arguments)
        {
            _returnType = returnType;
            _argumentTypes = argumentTypes;
            _arguments = arguments.ToArray();
        }

        public T GetArgument<T>(Op op, ParameterInfo parameter)
          where T : BaseFunction
        {
            return (T)_arguments[parameter.Index];
        }

        public T GetArgumentType<T>(Op op, ParameterInfo parameter)
            where T : IRType
        {
            if (op.GetType() == parameter.OwnerType)
            {
                return (T?)_argumentTypes[parameter.Index] ?? throw new InvalidOperationException("Run type infer first.");
            }
            else
            {
                throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
            }
        }

        public T GetReturnType<T>()
            where T : IRType
        {
            return (T?)_returnType ?? throw new InvalidOperationException("Run type infer first.");
        }
    }

    private sealed class FusionGraphCostVisitor : ExprVisitor<Cost, IRType>
    {
        protected override Cost VisitLeafVar(Var var)
        {
            return new Cost()
            {
                [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess((TensorType)var.CheckedType!),
            };
        }

        protected override Cost DefaultVisitLeaf(Expr expr)
        {
            return Cost.Zero;
        }

        protected override Cost VisitLeafCall(Call call)
        {
            Cost cost;
            if (call.Target is Op op)
            {
                var context = new GraphOpCostEvaluateContext(call.CheckedType, call.Arguments.AsValueEnumerable().Select(p => p.CheckedType).ToArray(), call.Arguments);
                cost = CompilerServices.EvaluateOpCost(op, context) ?? Cost.Zero;
            }
            else
            {
                throw new NotSupportedException();
            }

            return cost;
        }

        protected override Cost VisitLeafFusion(Fusion fusion)
        {
            var cost = new Cost()
            {
                [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess((TensorType)fusion.Body.CheckedType!),
            };
            cost += fusion.Parameters.AsValueEnumerable().Select(Visit).Sum() ?? Cost.Zero;
            return cost;
        }
    }
}

internal sealed class SingleInputFusionMergeRule : IRewriteRule
{
    private readonly Dictionary<int, Call> _mergedCache = new();

    public IPattern Pattern { get; } =
      IsCall(
        "caller",
        IsFusion("caller_fusion", Callable.StackVMModuleKind, IsWildcard(), IsVArgs(IsVar())),
        IsCall(
          "callee",
          IsFusion("callee_fusion", Callable.StackVMModuleKind, IsWildcard(), IsVArgs(IsVar())),
          IsWildcard("callee_input")));

    public static Fusion MergeSingleInputFusion(Call caller, Call callee, Fusion caller_fusion, Fusion callee_fusion, RunPassContext passOptions)
    {
        if (callee_fusion.Parameters.Length != 1 || caller_fusion.Parameters.Length != 1)
        {
            throw new NotSupportedException("Not Support Multi Inputs Fusion Merge");
        }

        // 1. replace the caller_fusion input_var with the callee_fusion body
        var new_fusion_body = Mutator.Substitute(e => object.ReferenceEquals(e, caller_fusion.Parameters[0]) ? callee_fusion.Body : null)().Rewrite(caller_fusion.Body);

        // 2. fold the store load
        // new_fusion_body = CompilerServices.Rewrite(new_fusion_body, new[] { new Passes.Rules.K510.FoldStoreLoad() }, passOptions.IndentDir("MergeSingleInputFusion"));
        var new_fusion = new Fusion($"{caller_fusion.Name}_{callee_fusion.Name}", Callable.StackVMModuleKind, new_fusion_body, callee_fusion.Parameters);

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
        var new_fusion_body = Mutator.Substitute(e =>
        {
            if (object.ReferenceEquals(e, caller_fusion.Parameters[0]))
            {
                return lhs_callee_fusion.Body;
            }

            if (object.ReferenceEquals(e, caller_fusion.Parameters[1]))
            {
                return rhs_callee_fusion.Body;
            }

            return null;
        })().Rewrite(caller_fusion.Body);

        // 2. fold the store load
        // new_fusion_body = CompilerServices.Rewrite(new_fusion_body, new[] { new Passes.Rules.K510.FoldStoreLoad() }, passOptions.IndentDir("MergeSingleInputFusion"));
        var new_fusion = new Fusion($"{caller_fusion.Name}_{lhs_callee_fusion.Name}_{rhs_callee_fusion.Name}", Callable.StackVMModuleKind, new_fusion_body, lhs_callee_fusion.Parameters);

        return new_fusion;
    }

    public Expr? GetReplace(IMatchResult result, RunPassContext options)
    {
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
