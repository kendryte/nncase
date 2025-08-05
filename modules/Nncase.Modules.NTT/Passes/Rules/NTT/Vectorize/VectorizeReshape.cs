// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

[RuleGenerator]
public sealed partial class VectorizeReshapePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsPack(
            "vectorize",
            "caller",
            _ => true,
            IsReshape(
                "reshape",
                "callee",
                _ => true,
                IsWildcard("input"),
                IsRankedShape("newShape")));

    private Expr? GetReplace(Pack vectorize, Call caller, Call callee, Expr input, RankedShape newShape)
    {
        var inShape = input.CheckedShape;
        var maxInputShape = CompilerServices.GetMaxShape(inShape);
        var maxNewShape = CompilerServices.GetMaxShape(newShape);
        if (!IRUtility.TryGetShapeMapMatrix(maxInputShape, maxNewShape, out var mat))
        {
            return null;
        }

        // TODO: more complex case
        var (forwardDict, backwardDict) = IRUtility.ShapeMapMatrixAsCompleteDict(mat);
        var vectorizeAxes = new List<int>();
        var vectorizeLanes = new List<int>();
        var rewritedNewShape = newShape.ToArray();
        for (int i = 0; i < vectorize.Axes.Count; i++)
        {
            var axis = vectorize.Axes[i];
            var lanes = vectorize.Lanes[i];

            // 1. [1024] -> pack([8, 128], [1], [8]): <8>[128] -> <8>[8, 16]
            foreach ((var inAxis, var newAxes) in forwardDict)
            {
                var vectorizeAxisIndex = newAxes.IndexOf(axis);
                if (vectorizeAxisIndex >= 0)
                {
                    if (vectorizeAxisIndex == newAxes.Count - 1
                        || newAxes.Skip(vectorizeAxisIndex + 1).All(x => x == 1))
                    {
                        // last axis, just use the vectorize axis
                        vectorizeAxes.Add(inAxis);
                        vectorizeLanes.Add(lanes);
                        rewritedNewShape[axis] /= lanes;
                    }
                    else
                    {
                        // We doesn't support vectorize axis in the middle of new shape
                        return null;
                    }
                }
            }

            // 2. [8, 128] -> pack([1024], [0], [8]): <8>[8, 16] -> <8>[128]
            if (backwardDict.TryGetValue(axis, out var inAxes))
            {
                // Find appropriate input axis to vectorize
                bool found = false;
                foreach (var inAxis in Enumerable.Reverse(inAxes))
                {
                    if (vectorizeAxes.Contains(inAxis))
                    {
                        // Already vectorized this axis
                        found = true;
                        continue;
                    }

                    if (inShape[inAxis] != 1)
                    {
                        if (Dimension.TryDivExactly(inShape[inAxis], lanes, out var newDim))
                        {
                            // found a valid axis
                            vectorizeAxes.Add(inAxis);
                            vectorizeLanes.Add(lanes);
                            rewritedNewShape[axis] = newDim;
                            found = true;
                            break;
                        }
                        else
                        {
                            // Dimension cannot be divided exactly, we cannot vectorize this axis
                            return null;
                        }
                    }
                }

                if (!found)
                {
                    // No valid axis found, we cannot vectorize this axis
                    return null;
                }
            }
        }

        return IR.F.Tensors.Reshape(
            IR.F.Tensors.Pack(input, vectorizeLanes.ToArray(), vectorizeAxes.ToArray()),
            new RankedShape(rewritedNewShape));
    }
}

[RuleGenerator]
public sealed partial class ReshapeDevectorizePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsReshape(
            "reshape",
            "caller",
            _ => true,
            PatternMatch.F.Tensors.IsUnpack(
                "devectorize",
                "callee",
                _ => true,
                IsWildcard("input")),
            IsRankedShape("newShape"));

    private Expr? GetReplace(Unpack devectorize, Call caller, Call callee, Expr input, RankedShape newShape)
    {
        var maxDevectorizeedShape = CompilerServices.GetMaxShape(callee.CheckedShape);
        var maxNewShape = CompilerServices.GetMaxShape(newShape);
        if (!IRUtility.TryGetShapeMapMatrix(maxDevectorizeedShape, maxNewShape, out var mat))
        {
            return null;
        }

        // TODO: more complex case
        var (forwardDict, backwardDict) = IRUtility.ShapeMapMatrixAsCompleteDict(mat);
        var devectorizeAxes = new List<int>();
        var devectorizeLanes = new List<int>();
        var rewritedNewShape = newShape.ToArray();
        for (int i = 0; i < devectorize.Axes.Count; i++)
        {
            var axis = devectorize.Axes[i];
            var lanes = devectorize.Lanes[i];

            // 1. unpack(<8>[128], [0], [8]) -> [8, 128]: <8>[128] -> <8>[8, 16]
            if (forwardDict.TryGetValue(axis, out var newAxes))
            {
                // Find appropriate new axis to vectorize
                bool found = false;
                foreach (var newAxis in Enumerable.Reverse(newAxes))
                {
                    if (newShape[newAxis] != 1)
                    {
                        if (Dimension.TryDivExactly(newShape[newAxis], lanes, out var newDim))
                        {
                            // found a valid axis
                            devectorizeAxes.Add(newAxis);
                            devectorizeLanes.Add(lanes);
                            rewritedNewShape[newAxis] = newDim;
                            found = true;
                            break;
                        }
                        else
                        {
                            // Dimension cannot be divided exactly, we cannot vectorize this axis
                            return null;
                        }
                    }
                }

                if (!found)
                {
                    // No valid axis found, we cannot vectorize this axis
                    return null;
                }
            }

            // 2. unpack(<8>[8, 16], [1], [8]) -> [1024]: <8>[8, 16] -> <8>[128]
            foreach ((var newAxis, var inAxes) in backwardDict)
            {
                if (devectorizeAxes.Contains(newAxis))
                {
                    // Already vectorized this axis
                    continue;
                }

                var vectorizeAxisIndex = inAxes.IndexOf(axis);
                if (vectorizeAxisIndex >= 0)
                {
                    if (vectorizeAxisIndex == inAxes.Count - 1
                        || inAxes.Skip(vectorizeAxisIndex + 1).All(x => x == 1))
                    {
                        // last axis, just use the vectorize axis
                        devectorizeAxes.Add(newAxis);
                        devectorizeLanes.Add(lanes);
                        rewritedNewShape[newAxis] /= lanes;
                    }
                    else
                    {
                        // We doesn't support vectorize axis in the middle of new shape
                        return null;
                    }
                }
            }
        }

        return IR.F.Tensors.Unpack(
            IR.F.Tensors.Reshape(input, new RankedShape(rewritedNewShape)),
            devectorizeLanes.ToArray(),
            devectorizeAxes.ToArray());
    }
}
