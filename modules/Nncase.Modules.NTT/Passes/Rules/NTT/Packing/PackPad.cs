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
public sealed partial class PackPadPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
    PatternMatch.F.Tensors.IsPack(
            "pack",
            "caller",
            _ => true,
            IsPad("pad", "callee", _ => true, IsWildcard("input"), IsPaddings("pads"), IsTensorConst("value")));

    private Expr? GetReplace(Pack pack, Call caller, Call callee, Expr input, Paddings pads, TensorConst value)
    {
        var newPads = pads.ToArray();
        for (var i = 0; i < pack.Axes.Count; i++)
        {
            var axis = pack.Axes[i];
            var lanes = pack.Lanes[i];

            // Make sure the padding of pack axis can be divided by pack lanes.
            if (Dimension.TryDivExactly(pads[axis].Before, lanes, out var before) &&
                Dimension.TryDivExactly(pads[axis].After, lanes, out var after))
            {
                newPads[axis] = (before, after);
            }
            else
            {
                return null; // Cannot pack this pad.
            }
        }

        return callee.WithArguments([
            (Pad.Input, caller.WithArguments([(Pack.Input, input)])),
            (Pad.Pads, new Paddings(newPads)),
        ]);
    }
}
