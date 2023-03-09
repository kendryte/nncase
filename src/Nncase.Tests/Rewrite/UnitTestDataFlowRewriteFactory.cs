// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.ReWriteTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestDataFlowRewriteFactory : TestClassBase
{
    public static TheoryData<IRewriteCase> DataOne => new()
    {
        new CombineClampAddMul(),
    };

    public static TheoryData<IRewriteCase> DataAll => new()
    {
        new MergeBinaryBeforeConv2DCase(),
        new ActivationsTransposePRelu(),
        new ActivationsTransposePRelu2(),
        new ActivationsTransposePRelu3(),
        new ActivationsTranspose(),
        new ActivationsTranspose2(),
        new PadTransposeCase(),
        new TransposeLeakyRelu(),
        new Conv2DPadsCase(),
        new ReduceWindow2DPadsCase(),
        new MobileNetV1TransposeCase(),
    };

    [Theory]
    [MemberData(nameof(DataOne))]
    public Task RunOneAsync(IRewriteCase @case) => RunCoreAsync(@case);

    [Theory]
    [MemberData(nameof(DataAll))]
    public Task RunAllAsync(IRewriteCase @case) => RunCoreAsync(@case);

    private async Task RunCoreAsync(IRewriteCase @case)
    {
        using var dumpScope = new DumpScope($"../{@case.Name}", DumpFlags.None);
        var pre = @case.PreExpr;
        CompilerServices.InferenceType(pre);
#if DEBUG
        DumpScope.Current.DumpIR(pre, "pre");
#endif
        var feed_dict = @case.FeedDict;
        IValue preValue, postValue;
        var preHashCode = pre.GetHashCode();
        using (var preScope = new DumpScope("Pre", DumpFlags.None))
        {
            preValue = pre.Body.Evaluate(feed_dict);
        }

        var pass = new DataflowPass { Name = "DataFlowOptimize" };
        foreach (var rule in @case.Rules)
        {
            pass.Add(rule);
        }

        var post = (Function)await pass.RunAsync(pre, new());
#if DEBUG
        DumpScope.Current.DumpIR(post, "post");
#endif
        Assert.NotEqual(preHashCode, post.GetHashCode());

        using (var postScope = new DumpScope("Post", DumpFlags.None))
        {
            postValue = post.Body.Evaluate(feed_dict);
        }

        Assert.True(Comparator.Compare(preValue, postValue));
    }
}
