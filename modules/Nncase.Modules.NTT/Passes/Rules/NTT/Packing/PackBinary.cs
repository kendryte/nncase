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
public sealed partial class VectorizeBinaryPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsVectorize(
            "vectorize",
            "caller",
            _ => true,
            IsBinary(
                "binary",
                "callee",
                _ => true,
                IsWildcard("lhs"),
                IsWildcard("rhs")));

    private Expr? GetReplace(Vectorize vectorize, Call callee, Expr lhs, Expr rhs)
    {
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var outputRank = callee.CheckedShape.Rank;

        var lhsVectorizedAxes = new List<int>();
        var rhsVectorizedAxes = new List<int>();
        var lhsLanes = new List<int>();
        var rhsLanes = new List<int>();

        for (int i = 0; i < vectorize.Axes.Count; i++)
        {
            var axis = vectorize.Axes[i];
            var lanes = vectorize.Lanes[i];

            if (!VectorizeUtility.TryPropagateArgument(outputRank, lhsShape, axis, lanes, lhsVectorizedAxes, lhsLanes))
            {
                return null; // Cannot vectorize lhs.
            }

            if (!VectorizeUtility.TryPropagateArgument(outputRank, rhsShape, axis, lanes, rhsVectorizedAxes, rhsLanes))
            {
                return null; // Cannot vectorize rhs.
            }
        }

        return callee.WithArguments([
            (Binary.Lhs, IR.F.Tensors.Vectorize(lhs, lhsLanes.ToArray(), lhsVectorizedAxes.ToArray())),
            (Binary.Rhs, IR.F.Tensors.Vectorize(rhs, rhsLanes.ToArray(), rhsVectorizedAxes.ToArray())),
        ]);
    }
}

[RuleGenerator]
public sealed partial class BinaryDevectorizeLhsPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
    IsBinary(
            "binary",
            "caller",
            _ => true,
            PatternMatch.F.Tensors.IsDevectorize("devectorize", "callee", _ => true, IsWildcard("lhs")),
            IsWildcard("rhs") with { TypePattern = !IsVector() });

    public static Expr? GetReplaceOperand(Devectorize devectorize, Call caller, Call callee, ParameterInfo devectorizedOperandParameter, ParameterInfo theOtherOperandParameter, Expr devectorizedOperand, Expr theOtherOperand)
    {
        var theOtherOperandShape = theOtherOperand.CheckedShape;
        var outputRank = callee.CheckedShape.Rank;

        var theOtherOperandVectorizedAxes = new List<int>();
        var theOtherOperandLanes = new List<int>();

        for (int i = 0; i < devectorize.Axes.Count; i++)
        {
            var axis = devectorize.Axes[i];
            var lanes = devectorize.Lanes[i];

            if (!VectorizeUtility.TryPropagateArgument(outputRank, theOtherOperandShape, axis, lanes, theOtherOperandVectorizedAxes, theOtherOperandLanes))
            {
                return null; // Cannot vectorize theOtherOperand.
            }
        }

        var devectorizedOperandExtend = outputRank - devectorizedOperand.CheckedShape.Rank;
        var newDevectorizeAxes = devectorize.Axes.Select(a => a + devectorizedOperandExtend).ToArray();

        return IR.F.Tensors.Devectorize(
            caller.WithArguments([
                (devectorizedOperandParameter, devectorizedOperand),
                (theOtherOperandParameter, IR.F.Tensors.Vectorize(theOtherOperand, theOtherOperandLanes.ToArray(), theOtherOperandVectorizedAxes.ToArray())),
            ]),
            devectorize.Lanes.ToArray(),
            newDevectorizeAxes);
    }

    private Expr? GetReplace(Devectorize devectorize, Call caller, Call callee, Expr lhs, Expr rhs)
    {
        return GetReplaceOperand(devectorize, caller, callee, Binary.Lhs, Binary.Rhs, lhs, rhs);
    }
}

[RuleGenerator]
public sealed partial class BinaryDevectorizeRhsPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
    IsBinary(
            "binary",
            "caller",
            _ => true,
            IsWildcard("lhs") with { TypePattern = !IsVector() },
            PatternMatch.F.Tensors.IsDevectorize("devectorize", "callee", _ => true, IsWildcard("rhs")));

    private Expr? GetReplace(Devectorize devectorize, Call caller, Call callee, Expr lhs, Expr rhs)
    {
        return BinaryDevectorizeLhsPropagation.GetReplaceOperand(devectorize, caller, callee, Binary.Rhs, Binary.Lhs, rhs, lhs);
    }
}
