using System.Collections.Generic;
using Nncase.IR;
using Nncase.Transform;
using Nncase.Transform.Mutators;
using Xunit;

namespace Nncase.Tests.ReWrite.FusionTest;

internal sealed class TestFusionGroupMutator : Transform.Mutators.FusionGroupMutator
{

    public TestFusionGroupMutator(IUsedByResult usedByAnalysisReslut, IEnumerable<IFusionMergeRule> preOrderfusionRules, IEnumerable<IFusionMergeRule> postOrderfusionRules, RunPassOptions passOptions) : base(usedByAnalysisReslut, preOrderfusionRules, postOrderfusionRules, passOptions)
    {
    }

    public override bool MergedFusionCheckCallBack(Fusion merged_fusion)
    {
        if (merged_fusion.Name.IndexOf("False") == -1)
            return true;
        return false;
    }
}


public class UnitTestDataFlowFusion : TestFixture.UnitTestFixtrue
{

    public static TheoryData<IDataFlowFusionCase> DataOne = new()
    {
     new  DataFlowType6_1FusionCaseLeft(),
     new  DataFlowType6_1FusionCaseRight(),
      new DataFlowType7FusionCaseLeft(),
      new DataFlowType7FusionCaseRight()
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

      new DataFlowType1FusionCaseRight(),
      new DataFlowType2FusionCaseRight(),
      new DataFlowType3FusionCaseRight(),
      new DataFlowType4FusionCaseRight(),
      new DataFlowType5FusionCaseRight(),
      new DataFlowType6FusionCaseRight(),
    };

    [Theory]
    [MemberData(nameof(DataOne))]
    public void RunOne(IDataFlowFusionCase fusionCase) => RunCore(fusionCase);

    public void RunCore(IDataFlowFusionCase fusionCase)
    {
        var passOptions = GetPassOptions(fusionCase.GetType().Name);
        var compileOptions = passOptions.CompileOptions;

        var target = CompilerServices.GetTarget(compileOptions.Target);
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var main = new Function(fusionCase.BuildBody(input), input);

        IRModule module = new(main);
        CompilerServices.InferenceType(main);
        CompilerServices.DumpIR(main, "pre", passOptions.DumpDir);

        var usedbyReslut = Analyser.AnalysisUsedBy(main);
        var mutator = new TestFusionGroupMutator(usedbyReslut,
          new IFusionMergeRule[]{
            new SameInputFusionMergeRule(),
          },
          new IFusionMergeRule[]{
            new SameInputFusionMergeRule(),
            new MultiInputFusionMergeRule(),
          },
         passOptions);
        var post = (Function)mutator.Visit(main);
        CompilerServices.DumpIR(post, "post", passOptions.DumpDir);

        var input_tensor = TestFixture.Testing.Rand<float>(1, 3, 224, 224);
        var feed_dict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance){
          { input, Value.FromTensor(input_tensor) }
        };

        var pre_result = CompilerServices.Evaluate(main.Body, feed_dict);
        var visitor = new FusionCounterVisitor();
        visitor.Visit(post.Body);
        Assert.Equal(fusionCase.FinalFusionCount, visitor.Count);
        var post_result = CompilerServices.Evaluate(post.Body, feed_dict);
        Assert.True(TestFixture.Comparator.AllEqual(pre_result, post_result));
    }

    [Theory]
    [MemberData(nameof(DataAll))]
    public void RunAll(IDataFlowFusionCase fusionCase) => RunCore(fusionCase);

}

internal sealed class FusionCounterVisitor : ExprVisitor<int, IRType>
{
    public int Count { get; private set; } = 0;

    public override int DefaultVisitLeaf(Expr expr) => 0;

    public override int VisitLeaf(Fusion expr)
    {
        Count++;
        return 0;
    }
}