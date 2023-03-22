// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.Passes.RulesFactory;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

public static partial class SimplifyFactory
{
    private static readonly ExprPattern X = IsWildcard();
    private static readonly ExprPattern Y = IsWildcard();
    private static readonly ExprPattern Z = IsWildcard();
    private static readonly ExprPattern W = IsWildcard();
    private static readonly ExprPattern U = IsWildcard();
    private static readonly ExprPattern V = IsWildcard();
    private static readonly ConstPattern C0 = IsConst();
    private static readonly ConstPattern C1 = IsConst();
    private static readonly ConstPattern C2 = IsConst();
    private static readonly ConstPattern C3 = IsConst();
    private static readonly ConstPattern C4 = IsConst();
    private static readonly ConstPattern C5 = IsConst();

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

    public static List<IRewriteRule> SimplifyAdd()
    {
        return _simplifyAdd;
    }

    public static List<IRewriteRule> SimplifyMul()
    {
        return _simplifyMul;
    }
}
