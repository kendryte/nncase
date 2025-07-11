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
                IsRankedShape("newShape")));

    private Expr? GetReplace(Pack pack, Call caller, Call callee, Expr input, RankedShape newShape)
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
        var packAxes = new List<int>();
        var packLanes = new List<int>();
        var rewritedNewShape = newShape.ToArray();
        for (int i = 0; i < pack.Axes.Count; i++)
        {
            var axis = pack.Axes[i];
            var lanes = pack.Lanes[i];

            // 1. [1024] -> pack([8, 128], [1], [8]): <8>[128] -> <8>[8, 16]
            foreach ((var inAxis, var newAxes) in forwardDict)
            {
                var packAxisIndex = newAxes.IndexOf(axis);
                if (packAxisIndex >= 0)
                {
                    if (packAxisIndex == newAxes.Count - 1
                        || newAxes.Skip(packAxisIndex + 1).All(x => x == 1))
                    {
                        // last axis, just use the pack axis
                        packAxes.Add(inAxis);
                        packLanes.Add(lanes);
                        rewritedNewShape[axis] /= lanes;
                    }
                    else
                    {
                        // We doesn't support pack axis in the middle of new shape
                        return null;
                    }
                }
            }

            // 2. [8, 128] -> pack([1024], [0], [8]): <8>[8, 16] -> <8>[128]
            if (backwardDict.TryGetValue(axis, out var inAxes))
            {
                // Find appropriate input axis to pack
                bool found = false;
                foreach (var inAxis in Enumerable.Reverse(inAxes))
                {
                    if (packAxes.Contains(inAxis))
                    {
                        // Already packed this axis
                        found = true;
                        continue;
                    }

                    if (inShape[inAxis] != 1)
                    {
                        if (Dimension.TryDivExactly(inShape[inAxis], lanes, out var newDim))
                        {
                            // found a valid axis
                            packAxes.Add(inAxis);
                            packLanes.Add(lanes);
                            rewritedNewShape[axis] = newDim;
                            found = true;
                            break;
                        }
                        else
                        {
                            // Dimension cannot be divided exactly, we cannot pack this axis
                            return null;
                        }
                    }
                }

                if (!found)
                {
                    // No valid axis found, we cannot pack this axis
                    return null;
                }
            }
        }

        return IR.F.Tensors.Reshape(
            IR.F.Tensors.Pack(input, packLanes.ToArray(), packAxes.ToArray()),
            new RankedShape(rewritedNewShape));
    }
}

[RuleGenerator]
public sealed partial class ReshapeUnpackPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsReshape(
            "reshape",
            "caller",
            _ => true,
            PatternMatch.F.Tensors.IsUnpack(
                "unpack",
                "callee",
                _ => true,
                IsWildcard("input")),
            IsRankedShape("newShape"));

    private Expr? GetReplace(Unpack unpack, Call caller, Call callee, Expr input, RankedShape newShape)
    {
        var maxUnpackedShape = CompilerServices.GetMaxShape(callee.CheckedShape);
        var maxNewShape = CompilerServices.GetMaxShape(newShape);
        if (!IRUtility.TryGetShapeMapMatrix(maxUnpackedShape, maxNewShape, out var mat))
        {
            return null;
        }

        // TODO: more complex case
        var (forwardDict, backwardDict) = IRUtility.ShapeMapMatrixAsCompleteDict(mat);
        var unpackAxes = new List<int>();
        var unpackLanes = new List<int>();
        var rewritedNewShape = newShape.ToArray();
        for (int i = 0; i < unpack.Axes.Count; i++)
        {
            var axis = unpack.Axes[i];
            var lanes = unpack.Lanes[i];

            // 1. unpack(<8>[128], [0], [8]) -> [8, 128]: <8>[128] -> <8>[8, 16]
            if (forwardDict.TryGetValue(axis, out var newAxes))
            {
                // Find appropriate new axis to pack
                bool found = false;
                foreach (var newAxis in Enumerable.Reverse(newAxes))
                {
                    if (newShape[newAxis] != 1)
                    {
                        if (Dimension.TryDivExactly(newShape[newAxis], lanes, out var newDim))
                        {
                            // found a valid axis
                            unpackAxes.Add(newAxis);
                            unpackLanes.Add(lanes);
                            rewritedNewShape[newAxis] = newDim;
                            found = true;
                            break;
                        }
                        else
                        {
                            // Dimension cannot be divided exactly, we cannot pack this axis
                            return null;
                        }
                    }
                }

                if (!found)
                {
                    // No valid axis found, we cannot pack this axis
                    return null;
                }
            }

            // 2. unpack(<8>[8, 16], [1], [8]) -> [1024]: <8>[8, 16] -> <8>[128]
            foreach ((var newAxis, var inAxes) in backwardDict)
            {
                if (unpackAxes.Contains(newAxis))
                {
                    // Already packed this axis
                    continue;
                }

                var packAxisIndex = inAxes.IndexOf(axis);
                if (packAxisIndex >= 0)
                {
                    if (packAxisIndex == inAxes.Count - 1
                        || inAxes.Skip(packAxisIndex + 1).All(x => x == 1))
                    {
                        // last axis, just use the pack axis
                        unpackAxes.Add(newAxis);
                        unpackLanes.Add(lanes);
                        rewritedNewShape[newAxis] /= lanes;
                    }
                    else
                    {
                        // We doesn't support pack axis in the middle of new shape
                        return null;
                    }
                }
            }
        }

        return IR.F.Tensors.Unpack(
            IR.F.Tensors.Reshape(input, new RankedShape(rewritedNewShape)),
            unpackLanes.ToArray(),
            unpackAxes.ToArray());
    }
}
