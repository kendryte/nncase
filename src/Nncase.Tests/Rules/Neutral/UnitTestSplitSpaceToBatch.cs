// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.NN;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestSpaceToBatch : TransformTestBase
{
    [Fact]
    public void TestSplitSpaceToBatch()
    {
        var i = SpaceToBatch(Testing.Rand<float>(1, 206, 192), new[] { 3 }, new[,] { { 0, 1 } });
        var originEvaluateResult = i.Evaluate();
        var newBody = TestMatched<SplitSpaceToBatch>(i);
        var ev = newBody.Evaluate();
        _ = Comparator.CosSimilarity(originEvaluateResult, ev);
        var dumpDir = Dumpper.Directory;
        var (_, kmodel) = Testing.BuildKModel("kmodel", new IRModule(new Function(newBody, System.Array.Empty<Var>())), CompileSession);
        var inputs = System.Array.Empty<Tensor>();
        var result = Testing.RunKModel(kmodel, dumpDir, inputs);
        var v = Comparator.CosSimilarity(ev, result);
        Assert.True(v[0] > 0.99f);
    }

    [Fact]
    public void TestSplitBatchToSpace()
    {
        var i = BatchToSpace(Testing.Rand<float>(3, 192, 67), new[] { 3 }, new[,] { { 0, 1 } });
        TestMatched<SplitBatchToSpace>(i);
    }
}
