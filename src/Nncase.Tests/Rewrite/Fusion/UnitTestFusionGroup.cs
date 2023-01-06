// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Nncase.IR;
using Nncase.Transform;
using Nncase.Transform.Mutators;
using Xunit;

namespace Nncase.Tests.ReWrite.FusionTest;

public class UnitTestFusionGroup : TestClassBase
{
    public static TheoryData<IDataFlowFusionCase> DataOne = new()
    {
        new DataFlowType9FusionCase(),
    };

    public static TheoryData<IDataFlowFusionCase> DataAll = new()
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

        new DataFlowType1FusionCaseRight(),
        new DataFlowType2FusionCaseRight(),
        new DataFlowType3FusionCaseRight(),
        new DataFlowType4FusionCaseRight(),
        new DataFlowType5FusionCaseRight(),
        new DataFlowType6FusionCaseRight(),
        new DataFlowType6_1FusionCaseRight(),
        new DataFlowType7FusionCaseRight(),
        new DataFlowType8FusionCase(),
    };

    [Theory]
    [MemberData(nameof(DataOne))]
    public void RunOne(IDataFlowFusionCase fusionCase) => RunCore(fusionCase);

    [Theory]
    [MemberData(nameof(DataAll))]
    public void RunAll(IDataFlowFusionCase fusionCase) => RunCore(fusionCase);

    private void RunCore(IDataFlowFusionCase fusionCase)
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var main = new Function(fusionCase.BuildBody(input), input);

        IRModule module = new(main);
        CompilerServices.InferenceType(main);

        var rewriter = new DataFlowMergeRewriter();
        var post = (Function)rewriter.Rewrite(main, new IMergeRewriteRule[]
        {
            new SameInputFusionMergeRule(),
            new MultiInputFusionMergeRule(),
            new ShortCutFusionMergeRule(),
        }, (usedby, rule, option) => new TestFusionGroupMutator(usedby, rule, option),
          new());

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
