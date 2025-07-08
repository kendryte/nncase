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
public sealed partial class PackReshapePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsPack(
            "pack",
            "caller",
            _ => true,
            IsReshape(
                "reshape",
                "callee",
                _ => true,
                IsWildcard("input"),
                IsFixedShape("newShape")));

    private Expr? GetReplace(Pack pack, Call caller, Call callee, Expr input, RankedShape newShape)
    {
        var maxInputShape = CompilerServices.GetMaxShape(input.CheckedShape);
        var maxNewShape = CompilerServices.GetMaxShape(newShape);
        if (!IRUtility.TryGetShapeMapMatrix(maxInputShape, maxNewShape, out var mat))
        {
            return null;
        }

        // TODO: more complex case
        var (forwardDict, backwardDict) = IRUtility.ShapeMapMatrixAsDict(mat);
        var packAxes = new int[input.CheckedShape.Rank];
        var packLanes = new int[input.CheckedShape.Rank];
        for (int i = 0; i < pack.Axes.Count; i++)
        {
            var a = pack.Axes[i];
            if (backwardDict[a].Count > 1)
            {
                return null;
            }
            else
            {
                packAxes[a] = backwardDict[a][0];
                packLanes[a] = pack.Lanes[i];
            }
        }

        if (packAxes.Distinct().Count() != packAxes.Length)
        {
            return null;
        }

        var ret = PackReshape.AddCandidate(input, newShape, forwardDict, backwardDict, packAxes, packLanes).FirstOrDefault();
        if (ret is not null)
        {
            return IR.F.Tensors.Pack(ret, pack.Lanes.ToArray(), pack.Axes.ToArray());
        }

        return null;
    }
}
