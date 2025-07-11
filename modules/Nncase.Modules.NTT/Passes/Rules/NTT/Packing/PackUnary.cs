// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Imaging;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

[RuleGenerator]
public sealed partial class PackUnaryPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsPack(
            "pack",
            "caller",
            _ => true,
            IsUnary(
                "unary",
                "callee",
                _ => true,
                IsWildcard("input")));

    private Expr? GetReplace(Call caller, Call callee, Expr input)
    {
        return callee.WithArguments([
            (Unary.Input, caller.WithArguments([(Pack.Input, input)])),
        ]);
    }
}

[RuleGenerator]
public sealed partial class UnaryUnpackPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsUnary(
            "unary",
            "caller",
            _ => true,
            PatternMatch.F.Tensors.IsUnpack(
                "unpack",
                "callee",
                _ => true,
                IsWildcard("input")));

    private Expr? GetReplace(Call caller, Call callee, Expr input)
    {
        return callee.WithArguments([
            (Unpack.Input, caller.WithArguments([(Unary.Input, input)])),
        ]);
    }
}

[RuleGenerator]
public sealed partial class SwishUnpackPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsSwish(
            "swish",
            "caller",
            _ => true,
            PatternMatch.F.Tensors.IsUnpack(
                "unpack",
                "callee",
                _ => true,
                IsWildcard("input")));

    private Expr? GetReplace(Call caller, Call callee, Expr input)
    {
        return callee.WithArguments([
            (Unpack.Input, caller.WithArguments([(Swish.Input, input)])),
        ]);
    }
}
