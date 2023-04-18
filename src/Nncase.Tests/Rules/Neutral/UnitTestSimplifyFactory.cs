// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Xunit;
using static Nncase.Passes.RulesFactory;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestSimplifyFactory
{
    [Fact]
    public void TestSimplifyFactory()
    {
        Assert.NotEqual(new(), SimplifyFactory.SimplifyAdd());
        Assert.NotEqual(new(), SimplifyFactory.SimplifyMul());
    }
}
