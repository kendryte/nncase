// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Diagnostics;
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
        CompileOptions.DumpFlags = DumpFlags.Evaluator;
        var i = SpaceToBatch(Testing.Rand<float>(1, 206, 192), new[] { 3 }, new[,] { { 0, 1 } });
        TestMatched<SplitSpaceToBatch>(i);
    }

    [Fact]
    public void TestSplitBatchToSpace()
    {
        var i = BatchToSpace(Testing.Rand<float>(3, 192, 67), new[] { 3 }, new[,] { { 0, 1 } });
        TestMatched<SplitBatchToSpace>(i);
    }
}
