// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using Microsoft.Extensions.DependencyInjection;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Analysis;
using Nncase.Passes.GraphPartition;
using Nncase.Passes.Mutators;
using QuikGraph;
using QuikGraph.Algorithms;
using Xunit;

namespace Nncase.Tests.ReWrite.FusionTest;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public class UnitTestFusionGroup : TestClassBase
{
    public static readonly TheoryData<IDataFlowFusionCase> DataOne = new()
    {
        new DataFlowType16FusionCase(),
    };

    public static readonly TheoryData<IDataFlowFusionCase> DataAll = new()
    {
        new DataFlowType0FusionCase(),
        new DataFlowType0NotFusionCase(),
        new DataFlowType1FusionCaseLeft(),
        new DataFlowType2FusionCaseLeft(),
        new DataFlowType3FusionCaseLeft(),
        new DataFlowType4FusionCaseLeft(),
        new DataFlowType5FusionCaseLeft(),
        new DataFlowType6FusionCaseLeft(),
        new DataFlowType6_1FusionCaseLeft(),
        new DataFlowType7FusionCaseLeft(),
        new DataFlowType10FusionCaseLeft(),
        new DataFlowType10_1FusionCaseLeft(),
        new DataFlowType11FusionCaseLeft(),
        new DataFlowType12FusionCaseLeft(),
        new DataFlowType13FusionCaseLeft(),
        new DataFlowType14FusionCaseLeft(),
        new DataFlowType1FusionCaseRight(),
        new DataFlowType2FusionCaseRight(),
        new DataFlowType3FusionCaseRight(),
        new DataFlowType4FusionCaseRight(),
        new DataFlowType5FusionCaseRight(),
        new DataFlowType6FusionCaseRight(),
        new DataFlowType6_1FusionCaseRight(),
        new DataFlowType7FusionCaseRight(),
        new DataFlowType8FusionCase(),
        new DataFlowType9FusionCase(),
        new DataFlowType10FusionCaseRight(),
        new DataFlowType10_1FusionCaseRight(),
        new DataFlowType11FusionCaseRight(),
        new DataFlowType12FusionCaseRight(),
        new DataFlowType13FusionCaseRight(),
        new DataFlowType14FusionCaseRight(),
    };

    public static readonly TheoryData<IDataFlowFusionCaseTwoStage> DataTwoStage = new()
    {
        new DataFlowType16FusionCase(),
        new DataFlowType15FusionCaseLeft(),
        new DataFlowType15FusionCaseRight(),
    };

    public UnitTestFusionGroup()
    {
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.Rewrite;
#endif
    }

    public IAnalyzerManager AnalyzerMananger => CompileSession.GetRequiredService<IAnalyzerManager>();

    [Fact]
    public void TestFusionMergeCandidateComparer()
    {
        var f1 = new Fusion("main", Callable.StackVMModuleKind, None.Default, Array.Empty<Var>());
        var f2 = new Fusion("main", Callable.StackVMModuleKind, None.Default, Array.Empty<Var>());
        var h1 = new HashSet<Fusion>() { f1, f2 };
        var h2 = new HashSet<Fusion>() { f1, f2 };
        Assert.Equal(FusionGroupMutator.GroupedMatchOptions.GetCandidateHashCode(h1), FusionGroupMutator.GroupedMatchOptions.GetCandidateHashCode(h2));
    }

    [Theory]
    [MemberData(nameof(DataOne))]
    public void RunOne(IDataFlowFusionCase fusionCase) => RunCore(fusionCase);

    [Theory]
    [MemberData(nameof(DataAll))]
    public void RunAll(IDataFlowFusionCase fusionCase) => RunCore(fusionCase);

    [Theory(Skip = "No Two Stage")]
    [MemberData(nameof(DataTwoStage))]
    public void TestTwoStage(IDataFlowFusionCaseTwoStage fusionCase)
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var main = new Function(fusionCase.BuildBody(input), input);

        IRModule module = new(main);
        CompilerServices.InferenceType(main);
#if DEBUG
        Dumpper.DumpDotIR(main, "pre");
#endif

        var input_tensor = Testing.Rand<float>(1, 3, 224, 224);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
          { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);

        var analysis = new Dictionary<System.Type, IAnalysisResult>
        {
            [typeof(IExprUserAnalysisResult)] = AnalyzerMananger.GetAnaylsis<IExprUserAnalysisResult>(main),
        };

        var preRewriter = new DataFlowMergeRewriter();
        var post = (Function)preRewriter.Rewrite(
            main,
            new IMergeRewriteRule[]
            {
                new ShortCutFusionMergeRuleLeft(),
                new ShortCutFusionMergeRuleRight(),
            },
            (rule, option) => new TestFusionGroupMutator(rule, option),
            new() { AnalysisResults = analysis, MatchOptions = new FusionGroupMutator.GroupedMatchOptions() });
#if DEBUG
        Dumpper.DumpDotIR(post, "post1");
#endif
        var visitor = new FusionCounterVisitor();
        visitor.Visit(post.Body);
        Assert.Equal(fusionCase.MidFusionCount, visitor.Count);

        var postRewriter = new DataFlowMergeRewriter();
        post = (Function)postRewriter.Rewrite(
            post,
            new IMergeRewriteRule[]
            {
                new SameInputFusionMergeRule(),
                new MultiInputFusionMergeRule(),
            },
            (rule, option) => new TestFusionGroupMutator(rule, option),
            new() { AnalysisResults = analysis });
#if DEBUG
        Dumpper.DumpDotIR(post, "post2");
#endif

        visitor = new FusionCounterVisitor();
        visitor.Visit(post.Body);
        Assert.Equal(fusionCase.FinalFusionCount, visitor.Count);
        var post_result = CompilerServices.Evaluate(post.Body, feed_dict);
        Assert.True(Comparator.AllEqual(pre_result, post_result));
    }

    private void RunCore(IDataFlowFusionCase fusionCase)
    {
        var caseName = fusionCase.GetType().Name;
        using var scope = new Diagnostics.DumpScope(caseName);
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var main = new Function(fusionCase.BuildBody(input), input);

        IRModule module = new(main);
        CompilerServices.InferenceType(main);
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Rewrite))
        {
            Diagnostics.DumpScope.Current.DumpDotIR(main, "pre");
        }

        var input_tensor = Testing.Rand<float>(1, 3, 224, 224);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
          { input, Value.FromTensor(input_tensor) },
        };
        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);

        Function post;
        {
            // 1. convert to quikgraph
            var biGraph = new BidirectionalGraph<ExprVertex, ExprEdge>(true);
            {
                var graphConvertor = new ExprGraphConvertor<ExprVertex, ExprEdge>();
                graphConvertor.Visit(main.Body, biGraph);
            }

            // 2. perform condensation
            var condenseAlgo = new CondensationGraphAlgorithm<ExprVertex, ExprEdge>(biGraph);
            condenseAlgo.IsEdgeCompatible += (algo, arg) =>
            {
                return (arg.Edge.Source.Expr, arg.Edge.Target.Expr) switch
                {
                    (Call { Target: Fusion { Name: string a } }, Call { Target: Fusion { Name: string b } }) => a.Contains("True", StringComparison.CurrentCulture) && b.Contains("True", StringComparison.CurrentCulture),
                    _ => false,
                };
            };

            condenseAlgo.IsGraphCompatible += (algo, edge) =>
            {
                return algo.CondensedGraph.IsDirectedAcyclicGraph();
            };

            condenseAlgo.Compute();

            if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Rewrite))
            {
                condenseAlgo.CondensedGraph.Dump($"{main.Name}Condensed", init => { });
                condenseAlgo.ClusteredGraph.Dump($"{main.Name}Cluster", algo =>
                {
                    algo.FormatVertex += (s, arg) =>
                    {
                        arg.VertexFormat.Label = $"{arg.Vertex.Expr.GetType().Name}";
                        if (arg.Vertex.Expr is Fusion f)
                        {
                            arg.VertexFormat.Label += " " + f.Name;
                        }
                    };
                });
            }

            // 3. reconstruction
            var constructor = new TestReconstructor(main.Name, main.ModuleKind, condenseAlgo);
            var postbody = constructor.Construct();
            post = main.With(body: postbody);
        }

        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Rewrite))
        {
            Diagnostics.DumpScope.Current.DumpDotIR(post, "post");
        }

        var visitor = new FusionCounterVisitor();
        visitor.Visit(post.Body);
        Assert.True(fusionCase.FinalFusionCount == visitor.Count, $"The TestCase {caseName} failed.");
        var post_result = CompilerServices.Evaluate(post.Body, feed_dict);
        Assert.True(Comparator.AllEqual(pre_result, post_result), $"The TestCase {caseName} failed.");
    }
}

internal sealed class TestFusionGroupMutator : Passes.Mutators.FusionGroupMutator
{
    public TestFusionGroupMutator(IMergeRewriteRule preOrderfusionRule, RunPassContext passOptions)
        : base(preOrderfusionRule, passOptions)
    {
    }

    public override bool MergedFusionCheckCallBack(Fusion merged_fusion, HashSet<Fusion> candidate_fusions)
    {
        if (!merged_fusion.Name.Contains("False", System.StringComparison.CurrentCulture))
        {
            return true;
        }

        return false;
    }
}

internal sealed class TestReconstructor : ExprReconstructor<ExprVertex, ExprEdge>
{
    public TestReconstructor(string funcName, string moduleKind, CondensationGraphAlgorithm<ExprVertex, ExprEdge> algo)
        : base(algo)
    {
        FuncName = funcName;
        ModuleKind = moduleKind;
    }

    public string FuncName { get; }

    public string ModuleKind { get; }

    protected override Expr OnComplexCluster(ClusteredBidirectionalGraph<ExprVertex, ExprEdge> cluster, int sortIndex)
    {
        var pairs = GetClusterArgumentPairs(cluster);
        var paramDict = new Dictionary<Expr, Var>(ReferenceEqualityComparer.Instance);
        var extractDict = new Dictionary<Expr, Expr>(ReferenceEqualityComparer.Instance);
        var argumentDict = new Dictionary<Var, Expr>(ReferenceEqualityComparer.Instance);
        foreach (var (pre, post) in pairs)
        {
            if (pre is not (Call or Var or If))
            {
                continue;
            }

            Var @var;
            Expr extract;
            @var = new Var(pre.CheckedType);
            extract = @var;

            var added = paramDict.TryAdd(pre, @var);
            if (added)
            {
                extractDict.Add(pre, extract);
                argumentDict.Add(@var, post);
            }
        }

        var cloner = new ExprClusterCloner(extractDict);
        var outVertices = cluster.OutVertices(Algo.ClusteredGraph).ToArray();
        var clones = new List<Expr>();
        foreach (var outVertex in outVertices)
        {
            clones.Add(cloner.Clone(outVertex.Expr, default));
        }

        var cloned = PostProcess(clones);
        var fusion = new Fusion($"{FuncName}_{sortIndex}_kernel", ModuleKind, cloned, paramDict.Values.OfType<Var>().ToArray());
        return new Call(fusion, paramDict.Values.OfType<Var>().Select(v => argumentDict[v]).ToArray());
    }

    private Expr PostProcess(List<Expr> clones)
    {
        Expr PostProcessSingle(Expr cloned, out bool changed)
        {
            changed = false;
            switch (cloned)
            {
                case IR.Tuple tp:
                    var nFields = new List<Expr>();
                    foreach (var item in tp.Fields)
                    {
                        nFields.Add(PostProcessSingle(item, out var childChanged));
                        changed |= childChanged;
                    }

                    if (changed)
                    {
                        return new IR.Tuple(nFields.ToArray());
                    }
                    else
                    {
                        return tp;
                    }

                case Expr e when e.CheckedType is DistributedType d:
                    changed = true;
                    return IR.F.Distributed.Boxing(e, d.TensorType);
                default:
                    return cloned;
            }
        }

        if (clones.Count == 1)
        {
            return PostProcessSingle(clones[0], out _);
        }
        else
        {
            return new IR.Tuple(clones.Select(c => PostProcessSingle(c, out _)).ToArray());
        }
    }
}

internal sealed class FusionCounterVisitor : ExprWalker
{
    public int Count { get; private set; }

    protected override Unit VisitLeafFusion(Fusion expr)
    {
        Count++;
        return default;
    }
}
