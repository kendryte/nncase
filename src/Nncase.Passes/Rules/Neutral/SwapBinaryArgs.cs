// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using System;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Math;
using static Nncase.Passes.Utility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public partial class SwapBinaryArgs : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsBinary("bn", "bnCall",
        op => op.BinaryOp == BinaryOp.Add || op.BinaryOp == BinaryOp.Mul || op.BinaryOp == BinaryOp.Min ||
              op.BinaryOp == BinaryOp.Max,
        IsTensorConst("lhs"),
        IsWildcard("rhs"));

    Expr? GetReplace(Call bnCall, Expr lhs, Expr rhs)
    {
        return bnCall.With(arguments: new[] { rhs, lhs });
    }
}