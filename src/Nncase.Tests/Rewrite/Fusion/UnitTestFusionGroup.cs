// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Nncase.IR;
using Nncase.Transform;
using Nncase.Transform.Mutators;
using Xunit;

namespace Nncase.Tests.ReWrite.FusionTest;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public class UnitTestFusionGroup : TestClassBase
{
    public static readonly TheoryData<IDataFlowFusionCase> DataOne = new()
    {
        new DataFlowType14FusionCaseLeft(),
        new DataFlowType14FusionCaseRight(),
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
        new DataFlowType11FusionCaseLeft(),
        new DataFlowType12FusionCaseLeft(),
        new DataFlowType13FusionCaseLeft(),

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
        new DataFlowType11FusionCaseRight(),
        new DataFlowType12FusionCaseRight(),
        new DataFlowType13FusionCaseRight(),
    };

    public static readonly TheoryData<IDataFlowFusionCaseTwoStage> DataTwoStage = new()
    {
        new DataFlowType15FusionCaseLeft(),
        new DataFlowType15FusionCaseRight(),
    };

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

        var preRewriter = new DataFlowMergeRewriter();
        var post = (Function)preRewriter.Rewrite(
            main,
            new IMergeRewriteRule[]
            {
                new ShortCutFusionMergeRuleLeft(),
                new ShortCutFusionMergeRuleRight(),
            },
            (usedby, rule, option) => new TestFusionGroupMutator(usedby, rule, option),
            new());
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
            (usedby, rule, option) => new TestFusionGroupMutator(usedby, rule, option),
            new());
#if DEBUG
        Dumpper.DumpDotIR(post, "post2");
#endif

        var input_tensor = Testing.Rand<float>(1, 3, 224, 224);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
          { input, Value.FromTensor(input_tensor) },
        };

        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
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
            (usedby, rule, option) => new TestFusionGroupMutator(usedby, rule, option),
            new());
#if DEBUG
        Dumpper.DumpDotIR(post, "post");
#endif

        var input_tensor = Testing.Rand<float>(1, 3, 224, 224);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
          { input, Value.FromTensor(input_tensor) },
        };

        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var visitor = new FusionCounterVisitor();
        visitor.Visit(post.Body);
        Assert.Equal(fusionCase.FinalFusionCount, visitor.Count);
        var post_result = CompilerServices.Evaluate(post.Body, feed_dict);
        Assert.True(Comparator.AllEqual(pre_result, post_result));
    }
}

internal sealed class TestFusionGroupMutator : Transform.Mutators.FusionGroupMutator
{
    public TestFusionGroupMutator(IUsedByResult usedByAnalysisReslut, IMergeRewriteRule preOrderfusionRule, RunPassContext passOptions)
        : base(usedByAnalysisReslut, preOrderfusionRule, passOptions)
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

internal sealed class FusionCounterVisitor : ExprVisitor<int, IRType>
{
    public int Count { get; private set; }

    public override int DefaultVisitLeaf(Expr expr) => 0;

    public override int VisitLeaf(Fusion expr)
    {
        Count++;
        return 0;
    }
}
