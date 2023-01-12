// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Tests.TestFixture;
using Nncase.Transform;
using Xunit;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.ReWriteTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestEGraphRewriteFactory : TestClassBase
{
    public static TheoryData<IRewriteCase> DataOne => new()
    {
        new RemoveMarkerCaseEgraph(),
    };

    public static TheoryData<IRewriteCase> DataAll => new()
    {
        new ActivationsTransposePRelu(),
        new ActivationsTransposePRelu2(),
        new ActivationsTransposePRelu3(),
        new ActivationsTranspose(),
        new ActivationsTranspose2(),
        new PadTransposeCase(),
        new MobileNetV1TransposeCase(),
        new Conv2DPadsCase(),
        new ReduceWindow2DPadsCase(),
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
        new PadTransposeCaseEgraph(),
    };

    [Theory]
    [MemberData(nameof(DataOne))]
    public Task RunOneAsync(IRewriteCase @case) => RunCoreAsync(@case);

    [Theory]
    [MemberData(nameof(DataAll))]
    public Task RunAllAsync(IRewriteCase @case) => RunCoreAsync(@case);

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

    private async Task RunCoreAsync(IRewriteCase @case)
    {
        var pre = @case.PreExpr;
        var infered = pre.InferenceType();
        Assert.True(infered);
        var pass = new EGraphPass { Name = "EGraphOptimize" };
        foreach (var rule in @case.Rules)
        {
            pass.Add(rule);
        }

        var post = (Function)await pass.RunAsync(pre, new());
        Assert.True(post.InferenceType());
        _ = CountRunTicks(pre, @case.FeedDict, out var pre_ret);
        _ = CountRunTicks(post, @case.FeedDict, out var post_ret);
        Assert.True(Comparator.AllEqual(pre_ret, post_ret));

        // note the parallel test will cause the time count error.
        // Assert.True(pre_time >= post_time);
    }
}
