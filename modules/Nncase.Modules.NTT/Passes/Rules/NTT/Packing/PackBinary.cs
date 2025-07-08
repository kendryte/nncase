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
public sealed partial class PackBinaryPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsPack(
            "pack",
            "caller",
            _ => true,
            IsBinary(
                "binary",
                "callee",
                _ => true,
                IsWildcard("lhs"),
                IsWildcard("rhs")));

    private Expr? GetReplace(Pack pack, Call callee, Expr lhs, Expr rhs)
    {
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var outputRank = callee.CheckedShape.Rank;

        var lhsPackedAxes = new List<int>();
        var rhsPackedAxes = new List<int>();
        var lhsLanes = new List<int>();
        var rhsLanes = new List<int>();

        for (int i = 0; i < pack.Axes.Count; i++)
        {
            var axis = pack.Axes[i];
            var lanes = pack.Lanes[i];

            if (!PackUtility.TryPropagateArgument(outputRank, lhsShape, axis, lanes, lhsPackedAxes, lhsLanes))
            {
                return null; // Cannot pack lhs.
            }

            if (!PackUtility.TryPropagateArgument(outputRank, rhsShape, axis, lanes, rhsPackedAxes, rhsLanes))
            {
                return null; // Cannot pack rhs.
            }
        }

        return callee.WithArguments([
            (Binary.Lhs, IR.F.Tensors.Pack(lhs, lhsLanes.ToArray(), lhsPackedAxes.ToArray())),
            (Binary.Rhs, IR.F.Tensors.Pack(rhs, rhsLanes.ToArray(), rhsPackedAxes.ToArray())),
        ]);
    }
}
