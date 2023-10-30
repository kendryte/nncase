// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.ReWriteTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestEGraphRewriteFactory : TestClassBase
{
    public UnitTestEGraphRewriteFactory()
    {
#if DEBUG
        CompileOptions.DumpFlags = DumpFlags.PassIR | DumpFlags.EGraphCost | DumpFlags.Rewrite;
#endif
    }

    public static TheoryData<IRewriteCase> DataOne => new()
    {
        new PReluTransposeCase(),
    };

    public static TheoryData<IRewriteCase> DataAll => new()
    {
        new FoldConv2DBnCase(),
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
        new RemoveMarkerCaseEgraph(),
        new FoldLayerNormCase(),
        new FoldSwishCase(),
        new FoldGeluCase(),
        new FoldHardSwishCase(),
        new MatMulToConv2DCase(),
        new ReduceCase(),
        new BroadcastCase(),
        new CastCase(),
        new TileCase(),
        new StackCase(),
        new BitcastCase(),
        new SliceCase(),
        new TopKCase(),
        new GatherCase(),
        new GatherNDCase(),
        new FlattenCase(),
        new SplitCase(),
        new SqueezeCase(),
        new ConcatCase(),
        new UnsqueezeCase(),
        new ExpandCase(),
        new ShapeOfCase(),
        new ReverseSequenceCase(),
        new WhereCase(),
        new RangeCase(),
        new SizeOfCase(),
        new BatchToSpaceCase(),
        new L2NormalizationCase(),
        new OneHotCase(),
        new CeluCase(),
        new EluCase(),
        new SeluCase(),
        new HardmaxCase(),
        new HardSigmoidCase(),
        new LRNCase(),
        new SoftmaxCase(),
        new CumSumCase(),
        new LSTMCase(),
        new InstanceNormalizationCase(),
        new HardSwishCase(),
        new SoftplusCase(),
        new SoftsignCase(),
        new LpNormalizationCase(),
        new Conv2DTransposeCase(),
        new LogSoftmaxCase(),
        new CompareCase(),
        new FakeDequantizeCase(),
        new FakeQuantizeCase(),
        new ReduceArgCase(),
        new NormalLikeCase(),
        new UniformLikeCase(),
        new UniformCase(),
        new ResizeImageCase(),
        new ProdCase(),
        new MultiReshapeCase(),
        new PReluTransposeCase(),
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
        using var dumpScope = new DumpScope($"../{@case.Name}");
        IValue pre_ret, post_ret;
        var pre = @case.PreExpr;
        var infered = pre.InferenceType();
        Assert.True(infered);
#if DEBUG
        DumpScope.Current.DumpIR(pre, "pre");
#endif
        var feedDict = @case.FeedDict;
        _ = CountRunTicks(pre, feedDict, out pre_ret);

        var module = new IRModule(pre);
        var pmgr = CompileSession.CreatePassManager("pmgr");
        pmgr.AddWithName<EGraphRulesPass>("EGraphOptimize").Configure(p =>
        {
            foreach (var rule in @case.Rules)
            {
                p.Add(rule);
            }
        });

        await pmgr.RunAsync(module);
        var post = (Function)module.Entry!;
        Assert.True(post.InferenceType());

#if DEBUG
        DumpScope.Current.DumpIR(post, "post");
#endif
        Assert.True(@case.ChecksPostCallBack(post));
        _ = CountRunTicks(post, feedDict, out post_ret);
        Assert.True(Comparator.Compare(pre_ret, post_ret));

        // note the parallel test will cause the time count error.
        // Assert.True(pre_time >= post_time);
    }
}
