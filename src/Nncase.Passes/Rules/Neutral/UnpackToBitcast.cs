// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class UnpackToBitcast : IRewriteRule
{
    public IPattern Pattern { get; } =
        IsUnpack(
            target_name: "unpack",
            "call",
            _ => true,
            IsWildcard("input") with { TypePattern = HasRankedShape() });

    public Expr? GetReplace(Expr call, Unpack unpack, Expr input)
    {
        var rank = input.CheckedShape.Rank;

        // If unpack axes are all the same with the last rank of input,
        // we can swap the unpack with a bitcast.
        if (unpack.Axes.All(a => a == rank - 1))
        {
            return IR.F.Tensors.Bitcast(input, call.CheckedDataType);
        }

        return null;
    }
}
