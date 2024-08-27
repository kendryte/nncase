﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Imaging;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.CPU;

public abstract class PackRule : RewriteRule<Pattern>
{
    public PackRule(int rank, int lane)
    {
        Rank = rank;
        Lane = lane;
    }

    public int Lane { get; }

    public int Rank { get; }

    public override Expr? GetReplace(IMatchResult result, RunPassContext options) => throw new NotImplementedException();
}

public class PackSoftmax : PackRule
{
    public PackSoftmax(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsSoftmax(
      "target",
      IsWildcard("input") with { TypePattern = IsFloat() },
      IsWildcard("axis") with { TypePattern = IsIntegralScalar() });

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var input = (Expr)result["input"];
        var axis = ((TensorConst)result["axis"]).Value.ToScalar<int>();
        var inShape = input.CheckedShape.ToValueArray();

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packed = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, float.NegativeInfinity, out var pads), lanes, packedAxes);
            var softmax = IR.F.CPU.PackedSoftmax(packed, axis, packedAxes);
            if (softmax.CheckedType is not InvalidType)
            {
                var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(softmax, lanes, packedAxes), inShape, pads);
                rets.Add(post);
            }
        }

        for (int i = 0; i < input.CheckedShape.Count; i++)
        {
            AddCandidate(new[] { i }, new[] { Lane });
            for (int j = i + 1; j < input.CheckedShape.Count; j++)
            {
                if (Rank > 1)
                {
                    AddCandidate(new[] { i, j }, new[] { Lane, Lane });
                }
            }
        }

        return rets;
    }
}

public sealed class PackResizeImage : PackRule
{
    public PackResizeImage(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsResizeImage("target", op => op.TransformationMode == ImageResizeTransformationMode.Asymmetric && op.IsTFResize == false, IsWildcard("input"), IsWildcard("roi"), IsTensorConst("newSize"), IsTensorConst("cubicCoeffA"), IsTensorConst("excludeOutside"), IsTensorConst("extrapolationValue"));

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var roi = (Expr)result["roi"];
        if (roi is not None && roi.CheckedShape.Size != 0)
        {
            return null!;
        }

        var rets = new List<Expr>();
        var op = (IR.Imaging.ResizeImage)result["target"];
        var input = (Expr)result["input"];
        var newSize = ((TensorConst)result["newSize"]).Value.ToArray<int>();
        var inShape = input.CheckedShape.ToValueArray();

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packedInput = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

            var resized = IR.F.CPU.ResizeImage(packedInput, packedAxes, padsInput, newSize, op.ResizeMode, op.TransformationMode, op.NearestMode);

            if (resized.CheckedType is not InvalidType)
            {
                var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(resized, lanes, packedAxes), inShape, padsInput);
                rets.Add(post);
            }
        }

        AddCandidate(new[] { 1 }, new[] { Lane });
        return rets;
    }
}

public sealed class PackInstanceNorm : PackRule
{
    public PackInstanceNorm(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsInstanceNormalization(
      "target",
      _ => true,
      IsWildcard("input") with { TypePattern = IsFloat() },
      IsWildcard("scale") with { TypePattern = IsFloat() },
      IsWildcard("bias") with { TypePattern = IsFloat() },
      IsTensorConst("eps") with { TypePattern = IsFloat() });

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.NN.InstanceNormalization)result["target"];
        var input = (Expr)result["input"];
        var scale = (Expr)result["scale"];
        var bias = (Expr)result["bias"];
        var eps = ((TensorConst)result["eps"]).Value.ToScalar<float>();
        var inShape = input.CheckedShape.ToValueArray();
        var pshape = scale.CheckedShape.ToValueArray();

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packedInput = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

            var pAxes = packedAxes.Where(i => i == 1).Select(i => 0).ToArray();
            var packedScale = PackUtility.PadForPack(scale, pshape, pAxes, lanes, 0f, out var padsScale);
            if (pAxes.Length > 0)
            {
                packedScale = IR.F.CPU.Pack(packedScale, Enumerable.Repeat(Lane, pAxes.Length).ToArray(), pAxes);
            }

            var packedBias = PackUtility.PadForPack(bias, pshape, pAxes, lanes, 0f, out var padsBias);
            if (pAxes.Length > 0)
            {
                packedBias = IR.F.CPU.Pack(packedBias, Enumerable.Repeat(Lane, pAxes.Length).ToArray(), pAxes);
            }

            var layernorm = IR.F.CPU.InstacneNorm(packedInput, packedScale, packedBias, eps, packedAxes, padsInput);

            if (layernorm.CheckedType is not InvalidType)
            {
                var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(layernorm, lanes, packedAxes), inShape, padsInput);
                rets.Add(post);
            }
        }

        for (int i = 0; i < input.CheckedShape.Count; i++)
        {
            AddCandidate(new[] { i }, new[] { Lane });
            for (int j = i + 1; j < input.CheckedShape.Count; j++)
            {
                if (Rank > 1)
                {
                    AddCandidate(new[] { i, j }, new[] { Lane, Lane });
                }
            }
        }

        return rets;
    }
}

public sealed class PackLayerNorm : PackRule
{
    public PackLayerNorm(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsLayerNorm(
      "target",
      _ => true,
      IsWildcard("input") with { TypePattern = IsFloat() },
      IsWildcard("scale") with { TypePattern = IsFloat() },
      IsWildcard("bias") with { TypePattern = IsFloat() });

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.NN.LayerNorm)result["target"];
        var input = (Expr)result["input"];
        var scale = (Expr)result["scale"];
        var bias = (Expr)result["bias"];
        var inShape = input.CheckedShape.ToValueArray();
        var pshape = scale.CheckedShape.ToValueArray();

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packedInput = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

            var pAxes = packedAxes.Where(i => i >= op.Axis).Select(i => i - op.Axis).ToArray();
            var packedScale = PackUtility.PadForPack(scale, pshape, pAxes, lanes, 0f, out var padsScale);
            if (pAxes.Length > 0)
            {
                packedScale = IR.F.CPU.Pack(packedScale, Enumerable.Repeat(Lane, pAxes.Length).ToArray(), pAxes);
            }

            var packedBias = PackUtility.PadForPack(bias, pshape, pAxes, lanes, 0f, out var padsBias);
            if (pAxes.Length > 0)
            {
                packedBias = IR.F.CPU.Pack(packedBias, Enumerable.Repeat(Lane, pAxes.Length).ToArray(), pAxes);
            }

            var layernorm = IR.F.CPU.PackedLayerNorm(packedInput, packedScale, packedBias, op.Axis, op.Epsilon, op.UseMean, packedAxes, padsInput);

            if (layernorm.CheckedType is not InvalidType)
            {
                var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(layernorm, lanes, packedAxes), inShape, padsInput);
                rets.Add(post);
            }
        }

        for (int i = 0; i < input.CheckedShape.Count; i++)
        {
            AddCandidate(new[] { i }, new[] { Lane });
            for (int j = i + 1; j < input.CheckedShape.Count; j++)
            {
                if (Rank > 1)
                {
                    AddCandidate(new[] { i, j }, new[] { Lane, Lane });
                }
            }
        }

        return rets;
    }
}

public sealed class PackMatMul : PackRule
{
    public PackMatMul(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsMatMul(
      "target",
      IsWildcard("lhs") with { TypePattern = IsFloat() },
      IsWildcard("rhs") with { TypePattern = IsFloat() });

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var lhs = (Expr)result["lhs"];
        var rhs = (Expr)result["rhs"];
        var candidate = (Expr)result[Pattern];
        var lhsShape = lhs.CheckedShape.ToValueArray();
        var rhsShape = rhs.CheckedShape.ToValueArray();

        void AddCandidate(int[] lhsPackedAxes, int[] rhsPackedAxes, int[] lhsLanes, int[] rhsLanes)
        {
            var packedLhs = IR.F.CPU.Pack(PackUtility.PadForPack(lhs, lhsShape, lhsPackedAxes, lhsLanes, 0f, out var lhsPadNums), lhsLanes, lhsPackedAxes);
            var packedRhs = IR.F.CPU.Pack(PackUtility.PadForPack(rhs, rhsShape, rhsPackedAxes, rhsLanes, 0f, out var rhsPadNums), rhsLanes, rhsPackedAxes);

            // TODO: support padding
            if (lhsPadNums.Any(x => x > 0) || rhsPadNums.Any(x => x > 0))
            {
                return;
            }

            var matmul = IR.F.CPU.PackedMatMul(packedLhs, packedRhs, lhsPackedAxes, lhsPadNums, rhsPackedAxes, rhsPadNums);
            var lhsAlign = System.Math.Max(lhsShape.Length, rhsShape.Length) - lhsShape.Length;
            var rhsAlign = System.Math.Max(lhsShape.Length, rhsShape.Length) - rhsShape.Length;

            var mPackIndex = Array.IndexOf(lhsPackedAxes, lhsShape.Length - 2);
            var nPackIndex = Array.IndexOf(rhsPackedAxes, rhsShape.Length - 1);
            var unpackAxes = new List<int>();
            var unpadNums = new List<int>();
            var unpackLanes = new List<int>();
            if (mPackIndex != -1)
            {
                unpackAxes.Add(lhsAlign + lhsPackedAxes[mPackIndex]);
                unpadNums.Add(lhsPadNums[mPackIndex]);
                unpackLanes.Add(lhsAlign + lhsLanes[mPackIndex]);
            }

            if (nPackIndex != -1)
            {
                unpackAxes.Add(rhsAlign + rhsPackedAxes[nPackIndex]);
                unpadNums.Add(rhsPadNums[nPackIndex]);
                unpackLanes.Add(rhsAlign + rhsLanes[nPackIndex]);
            }

            Expr post = matmul;
            if (unpackAxes.Any())
            {
                post = PackUtility.SliceForPack(IR.F.CPU.Unpack(matmul, unpackLanes.ToArray(), unpackAxes.ToArray()), candidate.CheckedShape.ToValueArray(), unpadNums.ToArray());
            }

            rets.Add(post);
        }

        // pack A's k and B's k
        AddCandidate(new[] { lhsShape.Length - 1 }, new[] { rhsShape.Length - 2 }, new[] { Lane }, new[] { Lane });

        // only pack A's m
        AddCandidate(new[] { lhsShape.Length - 2 }, Array.Empty<int>(), new[] { Lane }, Array.Empty<int>());

        // only pack B's n
        AddCandidate(Array.Empty<int>(), new[] { rhsShape.Length - 1 }, Array.Empty<int>(), new[] { Lane });

        if (Rank > 1)
        {
            AddCandidate(new[] { lhsShape.Length - 2, lhsShape.Length - 1 }, new[] { rhsShape.Length - 2, rhsShape.Length - 1 }, new[] { Lane, Lane }, new[] { Lane, Lane });
            AddCandidate(new[] { lhsShape.Length - 2 }, new[] { rhsShape.Length - 1 }, new[] { Lane }, new[] { Lane });
            AddCandidate(new[] { lhsShape.Length - 2, lhsShape.Length - 1 }, new[] { rhsShape.Length - 2 }, new[] { Lane, Lane }, new[] { Lane });
            AddCandidate(new[] { lhsShape.Length - 1 }, new[] { rhsShape.Length - 2, rhsShape.Length - 1 }, new[] { Lane }, new[] { Lane, Lane });
        }

        return rets;
    }
}

public sealed class PackUnary : PackRule
{
    public PackUnary(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsUnary(
      "target",
      _ => true,
      IsWildcard("input") with { TypePattern = IsFloat() });

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.Math.Unary)result["target"];
        var input = (Expr)result["input"];
        var inShape = input.CheckedShape.ToValueArray();

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packedInput = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);
            var unary = IR.F.Math.Unary(op.UnaryOp, packedInput);
            if (unary.CheckedType is not InvalidType)
            {
                var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(unary, lanes, packedAxes), inShape, padsInput);
                rets.Add(post);
            }
        }

        for (int i = 0; i < input.CheckedShape.Count; i++)
        {
            AddCandidate(new[] { i }, new[] { Lane });
            for (int j = i + 1; j < input.CheckedShape.Count; j++)
            {
                if (Rank > 1)
                {
                    AddCandidate(new[] { i, j }, new[] { Lane, Lane });
                }
            }
        }

        return rets;
    }
}

public sealed class PackBinary : PackRule
{
    public PackBinary(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsBinary(
      "target",
      _ => true,
      IsWildcard("lhs") with { TypePattern = IsFloat() },
      IsWildcard("rhs") with { TypePattern = IsFloat() });

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.Math.Binary)result["target"];
        var lhs = (Expr)result["lhs"];
        var rhs = (Expr)result["rhs"];
        var candidate = (Expr)result[Pattern];
        var lhsShape = lhs.CheckedShape.ToValueArray();
        var rhsShape = rhs.CheckedShape.ToValueArray();

        void AddCandidate(int[] lhsPackedAxes, int[] rhsPackedAxes, int[] lhsLanes, int[] rhsLanes)
        {
            var packedLhs = IR.F.CPU.Pack(PackUtility.PadForPack(lhs, lhsShape, lhsPackedAxes, lhsLanes, 0f, out var lhsPadNums), lhsLanes, lhsPackedAxes);
            var packedRhs = IR.F.CPU.Pack(PackUtility.PadForPack(rhs, rhsShape, rhsPackedAxes, rhsLanes, 0f, out var rhsPadNums), rhsLanes, rhsPackedAxes);

            var binary = IR.F.CPU.PackedBinary(packedLhs, packedRhs, op.BinaryOp, lhsPackedAxes, lhsPadNums, rhsPackedAxes, rhsPadNums);
            if (binary.CheckedType is not InvalidType)
            {
                var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(binary, lhsLanes.Length >= rhsLanes.Length ? lhsLanes : rhsLanes, lhsPackedAxes.Length >= rhsPackedAxes.Length ? lhsPackedAxes : rhsPackedAxes), candidate.CheckedShape.ToValueArray(), lhsPackedAxes.Length >= rhsPackedAxes.Length ? lhsPadNums : rhsPadNums);
                rets.Add(post);
            }
        }

        foreach (var arr in new[] { GeneratePackAxes(lhsShape), GeneratePackAxes(rhsShape) }.CartesianProduct())
        {
            var lhsPackedAxes = arr.First();
            var rhsPackedAxes = arr.Skip(1).First();
            if (lhsPackedAxes.Length <= Rank && rhsPackedAxes.Length <= Rank)
            {
                AddCandidate(lhsPackedAxes, rhsPackedAxes, Enumerable.Repeat(Lane, lhsPackedAxes.Length).ToArray(), Enumerable.Repeat(Lane, rhsPackedAxes.Length).ToArray());
            }
        }

        return rets;
    }

    public IEnumerable<int[]> GeneratePackAxes(int[] shape)
    {
        if (shape.Length == 0 || (shape.Length == 1 && shape[0] == 1))
        {
            yield return Array.Empty<int>();
        }
        else
        {
            for (int i = 0; i < shape.Length; i++)
            {
                yield return new[] { i };
                for (int j = i + 1; j < shape.Length; j++)
                {
                    yield return new[] { i, j };
                }
            }
        }
    }
}

public sealed class PackSwish : PackRule
{
    public PackSwish(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsSwish(
      "target",
      IsWildcard("input") with { TypePattern = IsFloat() },
      IsTensorConst("beta") with { TypePattern = IsFloatScalar() });

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var input = (Expr)result["input"];
        var beta = ((TensorConst)result["beta"]).Value.ToScalar<float>();
        var inShape = input.CheckedShape.ToValueArray();

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packed = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var pads), lanes, packedAxes);
            var swish = IR.F.NN.Swish(packed, beta);
            var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(swish, lanes, packedAxes), inShape, pads);
            rets.Add(post);
        }

        for (int i = 0; i < input.CheckedShape.Count; i++)
        {
            AddCandidate(new[] { i }, new[] { Lane });
            for (int j = i + 1; j < input.CheckedShape.Count; j++)
            {
                if (Rank > 1)
                {
                    AddCandidate(new[] { i, j }, new[] { Lane, Lane });
                }
            }
        }

        return rets;
    }
}

public sealed class PackTranspose : PackRule
{
    public PackTranspose(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsTranspose(
      "trans",
      IsWildcard("input") with { TypePattern = IsFloat() },
      IsTensorConst("perm") with { TypePattern = IsIntegral() });

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

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
                var unpackLanes = lanes.Select(l => perm.IndexOf(l)).ToArray();
                bool swap = unpackAxes.Length == 2 && unpackAxes[0] > unpackAxes[1];
                if (swap)
                {
                    (unpackAxes[0], unpackAxes[1]) = (unpackAxes[1], unpackAxes[0]);
                    (pads[0], pads[1]) = (pads[1], pads[0]);
                    (unpackLanes[0], unpackLanes[1]) = (unpackLanes[1], unpackLanes[0]);
                }

                var newShape = perm.Select(i => inShape[i]).ToArray();
                rets.Add(PackUtility.SliceForPack(IR.F.CPU.Unpack(tarns, unpackLanes, unpackAxes), newShape, pads));
            }
        }

        for (int i = 0; i < input.CheckedShape.Count; i++)
        {
            AddCandidate(new[] { i }, new[] { Lane });
            for (int j = i + 1; j < input.CheckedShape.Count; j++)
            {
                if (Rank > 1)
                {
                    AddCandidate(new[] { i, j }, new[] { Lane, Lane });
                }
            }
        }

        return rets;
    }
}

public sealed class PackUnsqueeze : PackRule
{
    public PackUnsqueeze(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsUnsqueeze(
      "unsq",
      IsWildcard("input") with { TypePattern = IsFloat() },
      IsTensorConst("axes") with { TypePattern = IsIntegral() });

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

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

                rets.Add(PackUtility.SliceForPack(IR.F.CPU.Unpack(post, lanes, unpackAxes), outShape.ToArray(), pads));
            }
        }

        for (int i = 0; i < input.CheckedShape.Count; i++)
        {
            AddCandidate(new[] { i }, new[] { Lane });
            for (int j = i + 1; j < input.CheckedShape.Count; j++)
            {
                if (Rank > 1)
                {
                    AddCandidate(new[] { i, j }, new[] { Lane, Lane });
                }
            }
        }

        return rets;
    }
}

public sealed class PackConv2D : PackRule
{
    public PackConv2D(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsConv2D(
        "conv",
        conv => conv.PadMode == PadMode.Constant,
        IsWildcard("input"),
        IsWildcard("weights"),
        IsWildcard("bias"),
        IsTensorConst("stride"),
        IsTensorConst("padding"),
        IsTensorConst("dilation"),
        IsTensorConst("groups"),
        IsTensorConst("fusedClamp"));

    public static Expr AddCandidate(Expr input, Expr weights, Expr bias, int[] strides, int[] padding, int[] wShape, int[] outShape)
    {
        var col = IR.F.CPU.Im2col(input, new[] { wShape[2], wShape[3] }, strides, padding);
        var newW = IR.F.Tensors.Reshape(weights, new[] { wShape[0], wShape[1] * wShape[2] * wShape[3] });
        var matmul = IR.F.Tensors.MatMul(newW, col); // [oc, b*oh*ow]
        var newBias = IR.F.Tensors.Reshape(bias, new[] { wShape[0], 1 });
        var add = matmul + newBias;
        if (outShape[0] == 1)
        {
            return IR.F.Tensors.Reshape(add, outShape);
        }

        return IR.F.Tensors.Transpose(IR.F.Tensors.Reshape(add, new[] { outShape[1], outShape[0], outShape[2], outShape[3] }), new[] { 1, 0, 2, 3 });
    }

    public static Expr AddPackedCandidate(Expr input, Expr weights, Expr bias, int[] strides, int[] padding, int[] wShape, int[] outShape, int lane)
    {
        var col = IR.F.CPU.Im2col(IR.F.CPU.Pack(input, new[] { lane }, new[] { 1 }), new[] { wShape[2], wShape[3] }, strides, padding, new[] { 1 }, new[] { 0 });
        var newW = IR.F.Tensors.Reshape(IR.F.CPU.Pack(weights, new[] { lane }, new[] { 1 }), new[] { wShape[0], wShape[1] / lane * wShape[2] * wShape[3] });
        var matmul = IR.F.CPU.PackedMatMul(newW, col, new[] { 1 }, new[] { 0 }, new[] { 0 }, new[] { 0 }); // [oc, b*oh*ow]
        var newBias = IR.F.Tensors.Reshape(bias, new[] { wShape[0], 1 });
        var add = matmul + newBias;
        if (outShape[0] == 1)
        {
            return IR.F.Tensors.Reshape(add, outShape);
        }
        else
        {
            return IR.F.Tensors.Transpose(IR.F.Tensors.Reshape(add, new[] { outShape[1], outShape[0], outShape[2], outShape[3] }), new[] { 1, 0, 2, 3 });
        }
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

        var input = (Expr)result["input"];
        var weights = (Expr)result["weights"];
        var bias = (Expr)result["bias"];
        var strides = ((TensorConst)result["stride"]).Value.ToArray<int>();
        var padding = ((TensorConst)result["padding"]).Value.ToArray<int>();
        var dilation = ((TensorConst)result["dilation"]).Value.ToArray<int>();
        var groups = ((TensorConst)result["groups"]).Value.ToScalar<int>();
        var fusedClamp = ((TensorConst)result["fusedClamp"]).Value.ToArray<float>();
        var wShape = weights.CheckedShape.ToValueArray();
        var outShape = ((Expr)result[Pattern]).CheckedShape.ToValueArray();
        if (groups != 1 || wShape[1] % Lane != 0 || dilation[0] != 1 || dilation[1] != 1 || fusedClamp[0] != float.NegativeInfinity || fusedClamp[1] != float.PositiveInfinity)
        {
            return rets;
        }

        // only pack on in channels
        rets.Add(AddCandidate(input, weights, bias, strides, padding, wShape, outShape));
        rets.Add(AddPackedCandidate(input, weights, bias, strides, padding, wShape, outShape, Lane));
        return rets;
    }
}

public sealed class PackReshape : PackRule
{
    public PackReshape(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsReshape(
      "target",
      IsWildcard("input") with { TypePattern = IsFloat() },
      IsTensorConst("newShape") with { TypePattern = IsIntegral() });

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

        var input = (Expr)result["input"];
        var newShape = ((TensorConst)result["newShape"]).Value.ToArray<int>();
        var inShape = input.CheckedShape.ToValueArray();

        // 1. find the mapping transforms
        if (!PackUtility.TryGetShapeMapMatrix(inShape, newShape, out var mat))
        {
            return new List<Expr> { };
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
                    var inAxes = backwardDict[outAxis];
                    if (inAxes.Count == 1 || (inAxes[^1] == axis && inShape[axis] % Lane == 0))
                    {
                        unpackAxes.Add(outAxis);
                    }
                    else
                    {
                        return;
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
                rets.Add(PackUtility.SliceForPack(IR.F.CPU.Unpack(post, lanes, unpackAxes.ToArray()), newShape, pads));
            }
        }

        for (int i = 0; i < input.CheckedShape.Count; i++)
        {
            AddCandidate(new[] { i }, new[] { Lane });
            for (int j = i + 1; j < input.CheckedShape.Count; j++)
            {
                if (Rank > 1)
                {
                    AddCandidate(new[] { i, j }, new[] { Lane, Lane });
                }
            }
        }

        return rets;
    }
}

public sealed class PackSlice : PackRule
{
    public PackSlice(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsSlice(
      "target",
      IsWildcard("input") with { TypePattern = IsFloat() },
      IsTensorConst("begins") with { TypePattern = IsIntegral() },
      IsTensorConst("ends") with { TypePattern = IsIntegral() },
      IsTensorConst("axes") with { TypePattern = IsIntegral() },
      IsTensorConst("strides") with { TypePattern = IsIntegral() });

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

        var input = (Expr)result["input"];
        var begins = ((TensorConst)result["begins"]).Value.ToArray<long>();
        var ends = ((TensorConst)result["ends"]).Value.ToArray<long>();
        var axes = ((TensorConst)result["axes"]).Value.ToArray<long>();
        var strides = ((TensorConst)result["strides"]).Value.ToArray<long>();
        var inShape = input.CheckedShape.ToValueArray();
        var candidate = (Expr)result[Pattern];
        for (int i = 0; i < axes.Length; i++)
        {
            ends[i] = ends[i] switch
            {
                < 0 => inShape[axes[i]] + ends[i],
                int.MaxValue => inShape[axes[i]],
                long.MaxValue => inShape[axes[i]],
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
                rets.Add(PackUtility.SliceForPack(IR.F.CPU.Unpack(post, lanes, packAxes), candidate.CheckedShape.ToValueArray(), pads));
            }
        }

        for (int i = 0; i < input.CheckedShape.Count; i++)
        {
            AddCandidate(new[] { i }, new[] { Lane });
            for (int j = i + 1; j < input.CheckedShape.Count; j++)
            {
                if (Rank > 1)
                {
                    AddCandidate(new[] { i, j }, new[] { Lane, Lane });
                }
            }
        }

        return rets;
    }
}

[RuleGenerator]
public sealed partial class FoldPackUnpack : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = PatternMatch.F.CPU.IsPack("pack", "caller", _ => true, PatternMatch.F.CPU.IsUnpack("unpack", "callee", _ => true, IsWildcard("input")));

    private Expr? GetReplace(IR.CPU.Pack pack, IR.CPU.Unpack unpack, Expr input)
    {
        if (pack.Axes.SequenceEqual(unpack.Axes) && pack.Lanes.SequenceEqual(unpack.Lanes))
        {
            return input;
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class FoldPackConcatUnpack : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = PatternMatch.F.CPU.IsPack("pack", "caller", _ => true, PatternMatch.F.Tensors.IsConcat("concat", _ => true, IsTuple("tuple", IsVArgsRepeat("fileds", exprs =>
        {
            var patterns = new Pattern[exprs.Length];
            for (int i = 0; i < exprs.Length; i++)
            {
                patterns[i] = PatternMatch.F.CPU.IsUnpack($"unpack_{i}", $"callee_{i}", _ => true, IsWildcard($"input_{i}"));
            }

            return patterns;
        }))));

    private Expr? GetReplace(IR.CPU.Pack pack, IR.Tensors.Concat concat, IReadOnlyList<Expr> fileds, IMatchResult result)
    {
        var inputs = new Expr[fileds.Count];
        for (int i = 0; i < fileds.Count; i++)
        {
            var unpack = (IR.CPU.Unpack)result[$"unpack_{i}"];
            if (pack.Axes.SequenceEqual(unpack.Axes) && pack.Lanes.SequenceEqual(unpack.Lanes))
            {
                inputs[i] = (Expr)result[$"input_{i}"];
            }
            else
            {
                return null;
            }
        }

        return IR.F.Tensors.Concat(new IR.Tuple(inputs), concat.Axis);
    }
}
