// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public partial class SimplifyBinaryCeilDiv : RewriteRule<Pattern>
{

    public override Pattern Pattern { get; } = IsBinary(
        "binary",
        "binaryCall",
        x => x.BinaryOp is BinaryOp.CeilDiv,
        IsWildcard("lhs"),
        IsTensorConst("rhs"));

    private Expr? GetReplace(Binary binary, Expr lhs, TensorConst rhs)
    {
        if (rhs.Value.ToScalar<int>() == 1)
        {
            return lhs;
        }

        return null;
    }
}
