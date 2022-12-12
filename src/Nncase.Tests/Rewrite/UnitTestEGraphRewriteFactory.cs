using System.Collections.Generic;
using System.IO;
using Nncase.IR;
using Nncase.Transform;
using Xunit;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.ReWriteTest;

public sealed class UnitTestEGraphRewriteFactory : TestFixture.UnitTestFixtrue
{
    public static TheoryData<IRewriteCase> DataOne => new(){
      new Conv2DPadsCase(),
    };

    public static TheoryData<IRewriteCase> DataAll => new(){
      new MobileNetV1TransposeCase(),
      new TransposeLeakyRelu(),
      new FoldReshapeCase(),
      new FoldTransposePadCase(),
      new FoldNopClampCase(),
      new FoldNopReshapeCase(),
      new TransposeDemoCase(),
      new ClassicDemo(),
      new FoldNopTransposeCase3(),
      new FoldNopTransposeCase2(),
      new FoldNopTransposeCase1(),
      new FoldTransposeCase(),
    };

    [Theory]
    [MemberData(nameof(DataOne))]
    public void RunOne(IRewriteCase @case) => RunCore(@case);

    async void RunCore(IRewriteCase @case)
    {
        var caseOptions = GetPassOptions().IndentDir(@case.Name);
        var pre = @case.PreExpr;
        var infered = pre.InferenceType();
        CompilerServices.DumpIR(pre, "pre", caseOptions.DumpDir);
        Assert.True(infered);
        var pass = new EGraphPass("EGraphOptimize");
        pass.Add(@case.Rules);
        var post = (Function)await pass.RunAsync(pre, caseOptions);
        Assert.True(post.InferenceType());
        long pre_time = CountRunTicks(pre, @case.FeedDict, out var pre_ret);
        long post_time = CountRunTicks(post, @case.FeedDict, out var post_ret);
        Assert.True(TestFixture.Comparator.AllEqual(pre_ret, post_ret));
        // note the parallel test will cause the time count error.
        // Assert.True(pre_time >= post_time);
    }

    [Theory]
    [MemberData(nameof(DataAll))]
    public void RunAll(IRewriteCase @case) => RunCore(@case);

    private static long CountRunTicks(Function pre, IReadOnlyDictionary<Var, IValue> feed_dict, out IValue ret)
    {
        long pre_time;
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        stopwatch.Start();
        ret = pre.Body.Evaluate(feed_dict);
        stopwatch.Stop();
        pre_time = stopwatch.ElapsedTicks;
        return pre_time;
    }
}
