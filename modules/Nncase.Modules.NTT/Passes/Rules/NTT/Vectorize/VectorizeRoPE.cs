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
public sealed partial class VectorizeRoPEPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsPack(
            "vectorize",
            "caller",
            _ => true,
            IsRoPE(
                "rope",
                "callee",
                _ => true,
                IsWildcard("input"),
                IsWildcard("cos"),
                IsWildcard("sin")));

    private Expr? GetReplace(Call caller, Pack vectorize, Call cos, Call sin, Call callee, Expr input)
    {
        var outputShape = (RankedShape)caller.CheckedShape;
        var outputRank = outputShape.Rank;
        if (vectorize.Axes.Contains(outputRank - 1))
        {
            var lastDim = outputShape[^1];
            if (lastDim is not DimConst dc
                || dc.Value % 2 != 0)
            {
                // The last dimension must be a constant and even
                return null;
            }
        }

        var sinCosVectorizedAxes = new List<int>();
        var sinCosLanes = new List<int>();

        for (int i = 0; i < vectorize.Axes.Count; i++)
        {
            var axis = vectorize.Axes[i];
            var lanes = vectorize.Lanes[i];

            if (!VectorizeUtility.TryPropagateArgument(outputRank, cos.CheckedShape, axis, lanes, sinCosVectorizedAxes, sinCosLanes))
            {
                return null; // Cannot vectorize sincos.
            }
        }

        // [head, seq, dim] -> [head, dim, seq]
        var inputT = IR.F.Tensors.Transpose(caller.WithArguments([(Pack.Input, input)]), [0, 2, 1]);
        var cosT = IR.F.Tensors.Transpose(IR.F.Tensors.Pack(cos, sinCosLanes.ToArray(), sinCosVectorizedAxes.ToArray()), [1, 0]);
        var sinT = IR.F.Tensors.Transpose(IR.F.Tensors.Pack(sin, sinCosLanes.ToArray(), sinCosVectorizedAxes.ToArray()), [1, 0]);
        var outputT = IR.F.NTT.VectorizedRoPE(inputT, cosT, sinT);
        return IR.F.Tensors.Transpose(outputT, [0, 2, 1]);
    }
}
