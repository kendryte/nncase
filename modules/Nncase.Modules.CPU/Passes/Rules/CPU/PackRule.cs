// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Toolkit.HighPerformance;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.CPU;

public abstract class PackRule
{
    public const int Lane = 32;

    public abstract Pattern Pattern { get; }

    public abstract List<Expr> GetReplace(Expr candidate);
}

public sealed class PackTranspose : PackRule
{
    public override Pattern Pattern { get; } = IsTranspose(
      "trans",
      IsWildcard("input") with { TypePattern = IsFloat() },
      IsTensorConst("perm") with { TypePattern = IsIntegral() });

    public override List<Expr> GetReplace(Expr candidate)
    {
        var rets = new List<Expr>();
        if (!CompilerServices.TryMatchRoot(candidate, Pattern, out var result))
        {
            return rets;
        }

        var input = (Expr)result["input"];
        var perm = ((TensorConst)result["perm"]).Value.ToArray<int>();
        var inShape = input.CheckedShape.ToValueArray();

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packed = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var pads), lanes, packedAxes);

            var tarns = IR.F.CPU.PackedTranspose(packed, perm, packedAxes);
            if (tarns.CheckedType is not InvalidType)
            {
                var unpackAxes = packedAxes.Select(axis => perm.IndexOf(axis)).ToArray();
                bool swap = unpackAxes.Length == 2 && unpackAxes[0] > unpackAxes[1];
                if (swap)
                {
                    (unpackAxes[0], unpackAxes[1]) = (unpackAxes[1], unpackAxes[0]);
                    (pads[0], pads[1]) = (pads[1], pads[0]);
                }

                var newShape = perm.Select(i => inShape[i]).ToArray();
                rets.Add(PackUtility.SliceForPack(IR.F.CPU.Unpack(tarns, unpackAxes), newShape, pads));
            }
        }

        for (int i = 0; i < input.CheckedShape.Count; i++)
        {
            AddCandidate(new[] { i }, new[] { Lane });
            for (int j = i + 1; j < input.CheckedShape.Count; j++)
            {
                AddCandidate(new[] { i, j }, new[] { Lane, Lane });
            }
        }

        return rets;
    }
}

public sealed class PackUnsqueeze : PackRule
{
    public override Pattern Pattern { get; } = IsUnsqueeze(
      "unsq",
      IsWildcard("input") with { TypePattern = IsFloat() },
      IsTensorConst("axes") with { TypePattern = IsIntegral() });

    public override List<Expr> GetReplace(Expr candidate)
    {
        var rets = new List<Expr>();
        if (!CompilerServices.TryMatchRoot(candidate, Pattern, out var result))
        {
            return rets;
        }

        var input = (Expr)result["input"];
        var axes = ((TensorConst)result["axes"]).Value.ToArray<int>();
        var inShape = input.CheckedShape.ToValueArray();

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packed = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var pads), lanes, packedAxes);

            var post = IR.F.Tensors.Unsqueeze(packed, axes);
            if (post.CheckedType is not InvalidType)
            {
                var unpackAxes = packedAxes.Select(axis => axis + axes.Count(i => i <= axis)).ToArray();
                var outShape = inShape.ToList();
                foreach (var axis in axes)
                {
                    if (axis >= 0)
                    {
                        outShape.Insert(axis, 1);
                    }
                    else
                    {
                        var index = System.Math.Max(outShape.Count + axis + 1, 0);
                        outShape.Insert(index, 1);
                    }
                }

                rets.Add(PackUtility.SliceForPack(IR.F.CPU.Unpack(post, unpackAxes), outShape.ToArray(), pads));
            }
        }

        for (int i = 0; i < input.CheckedShape.Count; i++)
        {
            AddCandidate(new[] { i }, new[] { Lane });
            for (int j = i + 1; j < input.CheckedShape.Count; j++)
            {
                AddCandidate(new[] { i, j }, new[] { Lane, Lane });
            }
        }

        return rets;
    }
}

public sealed class PackReshape : PackRule
{
    public override Pattern Pattern { get; } = IsReshape(
      "target",
      IsWildcard("input") with { TypePattern = IsFloat() },
      IsTensorConst("newShape") with { TypePattern = IsIntegral() });

    public override List<Expr> GetReplace(Expr candidate)
    {
        var rets = new List<Expr>();
        if (!CompilerServices.TryMatchRoot(candidate, Pattern, out var result))
        {
            return rets;
        }

        var input = (Expr)result["input"];
        var newShape = ((TensorConst)result["newShape"]).Value.ToArray<int>();
        var inShape = input.CheckedShape.ToValueArray();

        // 1. find the mapping transforms
        if (!PackUtility.TryGetShapeMapMatrix(inShape, newShape, out var mat))
        {
            return new List<Expr> { candidate };
        }

        var (forwardDict, backwardDict) = PackUtility.ShapeMapMatrixAsDict(mat);

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            // 1. skip when the packedAxes will be split or merge.
            var unpackAxes = new List<int>();
            foreach (var axis in packedAxes)
            {
                var mapedOutAxes = forwardDict[axis];
                if (mapedOutAxes.Count > 1)
                {
                    // split to more dim.
                    if (mapedOutAxes.Count(i => newShape[i] != 1) > 1)
                    {
                        continue;
                    }
                    else
                    {
                        // unsqueeze.
                        var outAxis = mapedOutAxes.FirstOrDefault(i => newShape[i] != 1, mapedOutAxes.First());
                        if (backwardDict[outAxis].Count != 1)
                        {
                            continue;
                        }

                        unpackAxes.Add(outAxis);
                    }
                }
                else
                {
                    var outAxis = mapedOutAxes.First();

                    // when the outAxis is merged dim, only support no transpose order and no pad.
                    if (backwardDict[outAxis][^1] == axis && inShape[axis] % Lane == 0)
                    {
                        unpackAxes.Add(outAxis);
                    }
                }
            }

            var packed = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var pads), lanes, packedAxes);
            var packedNewShape = newShape.ToArray();
            foreach (var (lane, axis) in lanes.Zip(unpackAxes))
            {
                packedNewShape[axis] = MathUtility.CeilDiv(packedNewShape[axis], lane);
            }

            var post = IR.F.Tensors.Reshape(packed, packedNewShape);
            if (post.CheckedType is not InvalidType)
            {
                rets.Add(PackUtility.SliceForPack(IR.F.CPU.Unpack(post, unpackAxes.ToArray()), newShape, pads));
            }
        }

        for (int i = 0; i < input.CheckedShape.Count; i++)
        {
            AddCandidate(new[] { i }, new[] { Lane });
            for (int j = i + 1; j < input.CheckedShape.Count; j++)
            {
                AddCandidate(new[] { i, j }, new[] { Lane, Lane });
            }
        }

        return rets;
    }
}

public sealed class PackSlice : PackRule
{
    public override Pattern Pattern { get; } = IsSlice(
      "target",
      IsWildcard("input") with { TypePattern = IsFloat() },
      IsTensorConst("begins") with { TypePattern = IsIntegral() },
      IsTensorConst("ends") with { TypePattern = IsIntegral() },
      IsTensorConst("axes") with { TypePattern = IsIntegral() },
      IsTensorConst("strides") with { TypePattern = IsIntegral() });

    public override List<Expr> GetReplace(Expr candidate)
    {
        var rets = new List<Expr>();
        if (!CompilerServices.TryMatchRoot(candidate, Pattern, out var result))
        {
            return rets;
        }

        var input = (Expr)result["input"];
        var begins = ((TensorConst)result["begins"]).Value.ToArray<int>();
        var ends = ((TensorConst)result["ends"]).Value.ToArray<int>();
        var axes = ((TensorConst)result["axes"]).Value.ToArray<int>();
        var strides = ((TensorConst)result["strides"]).Value.ToArray<int>();
        var inShape = input.CheckedShape.ToValueArray();
        for (int i = 0; i < axes.Length; i++)
        {
            ends[i] = ends[i] switch
            {
                < 0 => inShape[axes[i]] + ends[i],
                int.MaxValue => ends[i],
                _ => ends[i],
            };
        }

        if (strides.Any(s => s != 1))
        {
            return rets;
        }

        void AddCandidate(int[] packAxes, int[] lanes)
        {
            var packedBegins = begins.ToArray();
            var packedEnds = ends.ToArray();
            for (int i = 0; i < packAxes.Length; i++)
            {
                var packAxis = packAxes[i];
                int j = axes.IndexOf(packAxis);

                // when the slice axis was packed, it must have no pad.
                if (j != -1)
                {
                    if (begins[j] % lanes[i] == 0 && ends[j] % lanes[i] == 0)
                    {
                        packedBegins[j] = begins[j] / lanes[i];
                        packedEnds[j] = ends[j] / lanes[i];
                    }
                    else
                    {
                        return;
                    }
                }
            }

            var packed = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packAxes, lanes, 0f, out var pads), lanes, packAxes);
            var post = IR.F.Tensors.Slice(packed, packedBegins, packedEnds, axes, strides);
            if (post.CheckedType is not InvalidType)
            {
                rets.Add(PackUtility.SliceForPack(IR.F.CPU.Unpack(post, packAxes), candidate.CheckedShape.ToValueArray(), pads));
            }
        }

        for (int i = 0; i < input.CheckedShape.Count; i++)
        {
            AddCandidate(new[] { i }, new[] { Lane });
            for (int j = i + 1; j < input.CheckedShape.Count; j++)
            {
                AddCandidate(new[] { i, j }, new[] { Lane, Lane });
            }
        }

        return rets;
    }
}
