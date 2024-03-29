// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.PatternMatch;

using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerTile : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsTile(
        IsWildcard("input") with { TypePattern = HasFixedShape() },
        IsTensorConst("repeats"));

    private Expr? GetReplace(Expr input, int[] repeats)
    {
        // TODO: split input
        if (input.CheckedShape.ToList()[0] != 1)
        {
            return null;
        }

        // TODO: confirm d dim.
        if (input.CheckedShape.Count < 5 && repeats[0] == 1)
        {
            var inRes = Squeeze(input, new[] { 0 });
            var inResO = new Var(inRes.CheckedType);
            var newRepeats = repeats[1..];

            var tile = new Call(new Fusion("ncnn", NcnnTile(inResO, newRepeats), new[] { inResO }), inRes);
            return Unsqueeze(tile, new[] { 0 });
        }

        // if repeats[0] != 1, means that input can't squeeze batchSize dim.
        return null;
    }
}
