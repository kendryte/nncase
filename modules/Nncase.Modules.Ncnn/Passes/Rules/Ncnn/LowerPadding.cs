// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.PatternMatch;

using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerPadding : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsPad(
        "pad",
        "padCall",
        x => true,
        IsWildcard("input"),
        IsTensorConst("pads"),
        IsTensorConst("padValue"));

    private Expr? GetReplace(IR.NN.Pad pad, Expr input, Expr pads, Expr padValue)
    {
        // TODO: split input
        if (input.CheckedShape.ToList()[0] != 1)
        {
            return null;
        }

        var inRes = Squeeze(input, new[] { 0 });
        var inResO = new Var(inRes.CheckedType);
        var paddings = pads.Evaluate().AsTensor().ToArray<int>();
        var (front, behind, top, bottom, left, right) =
            (paddings[2], paddings[3], paddings[4], paddings[5], paddings[6], paddings[7]);
        var padV = padValue.Evaluate().AsTensor().ToArray<float>()[0]; // TODO: IsScaler or not.
        int type = pad.PadMode switch
        {
            PadMode.Constant => 0,
            PadMode.Edge => 1,
            PadMode.Reflect => 2,
            _ => throw new NotImplementedException("Ncnn not support other pad mode"),
        };

        var erf = new Call(new Fusion("ncnn", NcnnPadding(inResO, top, bottom, left, right, type, padV, front, behind), new[] { inResO }), inRes);

        return Unsqueeze(erf, new[] { 0 });
    }
}
