// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Reactive;
using Microsoft.Extensions.DependencyInjection;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Analysis;
using Nncase.Passes.Mutators;
using Xunit;

namespace Nncase.Tests.ReWrite.FusionTest;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public class UnitTestFusionGroup : TestClassBase
{
    public static readonly TheoryData<IDataFlowFusionCase> DataOne = new()
    {
        // new DataFlowType16FusionCase(),
        new DataFlowType2FusionCaseLeft(),
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

    [Theory]
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

        var analysis = new Dictionary<Type, IAnalysisResult>
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

        var analysis = new Dictionary<Type, IAnalysisResult>
        {
            [typeof(IExprUserAnalysisResult)] = AnalyzerMananger.GetAnaylsis<IExprUserAnalysisResult>(main),
        };

        var rewriter = new DataFlowMergeRewriter();
        var post = (Function)rewriter.Rewrite(
            main,
            new IMergeRewriteRule[]
            {
                new SameInputFusionMergeRule(),
                new MultiInputFusionMergeRule(),
                new ShortCutFusionMergeRuleLeft(),
                new ShortCutFusionMergeRuleRight(),
            },
            (rule, option) => new TestFusionGroupMutator(rule, option),
            new() { AnalysisResults = analysis });
#if DEBUG
        Dumpper.DumpDotIR(post, "post");
#endif

        var visitor = new FusionCounterVisitor();
        visitor.Visit(post.Body);
        Assert.Equal(fusionCase.FinalFusionCount, visitor.Count);
        var post_result = CompilerServices.Evaluate(post.Body, feed_dict);
        Assert.True(Comparator.AllEqual(pre_result, post_result));
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

internal sealed class FusionCounterVisitor : ExprWalker
{
    public int Count { get; private set; }

    protected override Unit VisitLeafFusion(Fusion expr)
    {
        Count++;
        return default;
    }
}
