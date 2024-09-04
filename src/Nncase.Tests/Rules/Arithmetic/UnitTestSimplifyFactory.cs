// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Arithmetic;
using Nncase.PatternMatch;
using Xunit;
using static Nncase.Passes.RulesFactory;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.Rules.ArithmeticTest;

public class UnitTestSimplifyFactory
{
    private static readonly ExprPattern X = IsWildcard();
    private static readonly ConstPattern C0 = IsConst();
    private static readonly ConstPattern C1 = IsConst();

    private static readonly List<IRewriteRule> _simplifyAdd = new()
     {
         Rewrite(X + 0, X),
         Rewrite(0 + X, X),
         Rewrite(X + C0 + C1, X + (C0 + C1)),
         Rewrite(C0 + X + C1, X + (C0 + C1)),
         Rewrite(C1 + (X + C0), X + (C0 + C1)),
         Rewrite(C1 + (C0 + X), X + (C0 + C1)),
     };

    private static readonly List<IRewriteRule> _simplifyMul = new()
     {
         Rewrite(X * 1, X),
         Rewrite(1 * X, X),
     };

    [Fact]
    public void TestSimplifyFactory()
    {
        Assert.Equal(_simplifyAdd.ToString(), SimplifyFactory.SimplifyAdd().ToString());
        Assert.Equal(_simplifyMul.ToString(), SimplifyFactory.SimplifyMul().ToString());
    }
}
