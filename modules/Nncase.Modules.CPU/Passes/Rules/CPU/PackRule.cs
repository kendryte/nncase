// Copyright (c) Canaan Inc. All rights reserved.
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

public sealed class PackResizeImage : PackRule
{
    public PackResizeImage(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsResizeImage("target", op => op.TransformationMode == ImageResizeTransformationMode.Asymmetric && op.IsTFResize == false, IsWildcard("input") with { TypePattern = !IsVector() }, IsWildcard("roi"), IsTensorConst("newSize"), IsTensorConst("cubicCoeffA"), IsTensorConst("excludeOutside"), IsTensorConst("extrapolationValue"));

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

            var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(resized, lanes, packedAxes), inShape, padsInput);
            if (post.CheckedType is not InvalidType)
            {
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
      IsWildcard("input") with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("scale") with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("bias") with { TypePattern = IsFloat() & !IsVector() },
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

public sealed class PackMatMul : PackRule
{
    public PackMatMul(int rank, int lane)
        : base(rank, lane)
    {
    }

    [Flags]
    public enum PackKind : byte
    {
        None = 1 << 0,
        M = 1 << 1,
        K = 1 << 2,
        N = 1 << 3,
    }

    public override Pattern Pattern { get; } = IsMatMul(
      "target",
      IsWildcard("lhs", e => e is not Call { Target: IR.CPU.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("rhs", e => e is not Call { Target: IR.CPU.Unpack }) with { TypePattern = IsFloat() & !IsVector() });

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var lhs = (Expr)result["lhs"];
        var rhs = (Expr)result["rhs"];
        var candidate = (Expr)result[Pattern];
        var lhsShape = lhs.CheckedShape.ToValueArray();
        var rhsShape = rhs.CheckedShape.ToValueArray();
        var rcontext = new RuleContext(rets, lhs, rhs, candidate, lhsShape, rhsShape);

        // pack A's k and B's k
        AddCandidate(rcontext, PackKind.K, PackKind.K);

        // only pack A's m
        AddCandidate(rcontext, PackKind.M, PackKind.None);

        // only pack B's n
        AddCandidate(rcontext, PackKind.None, PackKind.N, transB: rhs is Const);
        if (Rank > 1)
        {
            // pack A's m and B's n, when B is const, force transpose
            AddCandidate(rcontext, PackKind.M, PackKind.N, transB: rhs is Const);

            // pack A's m,k and B's k,n
            AddCandidate(rcontext, PackKind.M | PackKind.K, PackKind.K | PackKind.N, transB: rhs is Const);

            // pack A's m,k and B's k
            AddCandidate(rcontext, PackKind.M | PackKind.K, PackKind.K);

            // pack A's k and B's k,n
            AddCandidate(rcontext, PackKind.K, PackKind.K | PackKind.N, transB: lhs is Const);
        }

        return rets;
    }

    private void AddCandidate(RuleContext context, PackKind lhsPack, PackKind rhsPack, bool transA = false, bool transB = false)
    {
        var (rets, lhs, rhs, candidate, _, _) = context;
        var lhsShape = context.LhsShape.ToArray();
        var rhsShape = context.RhsShape.ToArray();
        if (transA)
        {
            var perm = Enumerable.Range(0, lhsShape.Length).ToArray();
            (perm[^2], perm[^1]) = (perm[^1], perm[^2]);
            (lhsShape[^2], lhsShape[^1]) = (lhsShape[^1], lhsShape[^2]);
            lhs = IR.F.Tensors.Transpose(lhs, perm);
        }

        if (transB)
        {
            var perm = Enumerable.Range(0, rhsShape.Length).ToArray();
            (perm[^2], perm[^1]) = (perm[^1], perm[^2]);
            (rhsShape[^2], rhsShape[^1]) = (rhsShape[^1], rhsShape[^2]);
            rhs = IR.F.Tensors.Transpose(rhs, perm);
        }

        int[] lhsLanes;
        int[] lhsPackedAxes;
        var (lm, lk) = transA ? (lhsShape.Length - 1, lhsShape.Length - 2) : (lhsShape.Length - 2, lhsShape.Length - 1);
        var (rk, rn) = transB ? (rhsShape.Length - 1, rhsShape.Length - 2) : (rhsShape.Length - 2, rhsShape.Length - 1);
        switch (lhsPack)
        {
            case PackKind.None:
                lhsLanes = Array.Empty<int>();
                lhsPackedAxes = Array.Empty<int>();
                break;
            case PackKind.M:
                lhsLanes = [Lane];
                lhsPackedAxes = [lm];
                break;
            case PackKind.K:
                lhsLanes = [Lane];
                lhsPackedAxes = [lk];
                break;
            case PackKind.M | PackKind.K:
                lhsLanes = [Lane, Lane];
                lhsPackedAxes = [lm, lk];
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(lhsPack), lhsPack.ToString());
        }

        int[] rhsLanes;
        int[] rhsPackedAxes;
        switch (rhsPack)
        {
            case PackKind.None:
                rhsLanes = Array.Empty<int>();
                rhsPackedAxes = Array.Empty<int>();
                break;
            case PackKind.N:
                rhsLanes = [Lane];
                rhsPackedAxes = [rn];
                break;
            case PackKind.K:
                rhsLanes = [Lane];
                rhsPackedAxes = [rk];
                break;
            case PackKind.K | PackKind.N:
                rhsLanes = [Lane, Lane];
                rhsPackedAxes = [rk, rn];
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(rhsPack), rhsPack.ToString());
        }

        var packedLhs = IR.F.CPU.Pack(PackUtility.PadForPack(lhs, lhsShape, lhsPackedAxes, lhsLanes, 0f, out var lhsPadNums), lhsLanes, lhsPackedAxes);
        var packedRhs = IR.F.CPU.Pack(PackUtility.PadForPack(rhs, rhsShape, rhsPackedAxes, rhsLanes, 0f, out var rhsPadNums), rhsLanes, rhsPackedAxes);

        // TODO: support padding
        if (lhsPadNums.Any(x => x > 0) || rhsPadNums.Any(x => x > 0))
        {
            return;
        }

        var matmul = IR.F.CPU.PackedMatMul(packedLhs, packedRhs, lhsPackedAxes, lhsPadNums, rhsPackedAxes, rhsPadNums, transA, transB);

        var outRank = System.Math.Max(lhsShape.Length, rhsShape.Length);
        var lhsAlign = outRank - lhsShape.Length;
        var rhsAlign = outRank - rhsShape.Length;

        var unpackAxes = new List<int>();
        var unpadNums = new List<int>();
        var unpackLanes = new List<int>();
        if (lhsPack.HasFlag(PackKind.M))
        {
            var mPackIndex = Array.IndexOf(lhsPackedAxes, lm);
            unpackAxes.Add(outRank - 2);
            unpadNums.Add(lhsPadNums[mPackIndex]);
            unpackLanes.Add(Lane);
        }

        if (rhsPack.HasFlag(PackKind.N))
        {
            var nPackIndex = Array.IndexOf(rhsPackedAxes, rn);
            unpackAxes.Add(outRank - 1);
            unpadNums.Add(rhsPadNums[nPackIndex]);
            unpackLanes.Add(Lane);
        }

        Expr post = matmul;
        if (unpackAxes.Any())
        {
            post = PackUtility.SliceForPack(IR.F.CPU.Unpack(matmul, unpackLanes.ToArray(), unpackAxes.ToArray()), candidate.CheckedShape.ToValueArray(), unpadNums.ToArray());
        }

        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }
    }

    private sealed record RuleContext(List<Expr> Results, Expr Lhs, Expr Rhs, Expr Candidate, IReadOnlyList<int> LhsShape, IReadOnlyList<int> RhsShape)
    {
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
      IsWildcard("input", e => e is not Call { Target: IR.CPU.Unpack }) with { TypePattern = IsFloat() & !IsVector() });

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
            var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(unary, lanes, packedAxes), inShape, padsInput);
            if (unary.CheckedType is not InvalidType)
            {
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
      IsWildcard("lhs", e => e is not Call { Target: IR.CPU.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("rhs", e => e is not Call { Target: IR.CPU.Unpack }) with { TypePattern = IsFloat() & !IsVector() });

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
            var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(binary, lhsLanes.Length >= rhsLanes.Length ? lhsLanes : rhsLanes, lhsPackedAxes.Length >= rhsPackedAxes.Length ? lhsPackedAxes : rhsPackedAxes), candidate.CheckedShape.ToValueArray(), lhsPackedAxes.Length >= rhsPackedAxes.Length ? lhsPadNums : rhsPadNums);
            if (post.CheckedType is not InvalidType)
            {
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
      IsWildcard("input", e => e is not Call { Target: IR.CPU.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
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
            if (post.CheckedType is not InvalidType)
            {
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

public sealed class PackTranspose : PackRule
{
    public PackTranspose(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsTranspose(
      "trans",
      IsWildcard("input", e => e is not Call { Target: IR.CPU.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
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

            var tarns = IR.F.Tensors.Transpose(packed, perm);
            if (tarns.CheckedType is not InvalidType)
            {
                var partialPerm = perm.Select(axis => packedAxes.IndexOf(axis)).Where(x => x != -1).ToArray();
                var unpackAxes = packedAxes.Select(axis => perm[axis]).ToArray();
                var unpackPads = Enumerable.Range(0, pads.Length).Select(i => pads[partialPerm[i]]).ToArray();
                var unpackLanes = Enumerable.Range(0, lanes.Length).Select(i => lanes[partialPerm[i]]).ToArray();
                var newShape = perm.Select(i => inShape[i]).ToArray();
                var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(tarns, unpackLanes, unpackAxes), newShape, unpackPads);
                if (post.CheckedType is not InvalidType)
                {
                    rets.Add(post);
                }
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
      IsWildcard("input", e => e is not Call { Target: IR.CPU.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
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
            var unseq = IR.F.Tensors.Unsqueeze(packed, axes);
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

            var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(unseq, lanes, unpackAxes), outShape.ToArray(), pads);
            if (post.CheckedType is not InvalidType)
            {
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

public sealed class PackConv2D : PackRule
{
    public PackConv2D(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsConv2D(
        "conv",
        conv => conv.PadMode == PadMode.Constant,
        IsWildcard("input", e => e is not Call { Target: IR.CPU.Unpack }) with { TypePattern = !IsVector() },
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
      IsWildcard("input", e => e is not Call { Target: IR.CPU.Unpack }) with { TypePattern = !IsVector() },
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
                    if (mapedOutAxes.Count(i => newShape[i] != 1) > 1)
                    {
                        // we can pack on split axis and unpack on splited last axis.
                        unpackAxes.Add(mapedOutAxes[^1]);
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

            var nreshape = IR.F.Tensors.Reshape(packed, packedNewShape);
            var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(nreshape, lanes, unpackAxes.ToArray()), newShape, pads);
            if (post.CheckedType is not InvalidType)
            {
                rets.Add(post);
            }
        }

        for (int i = 0; i < input.CheckedShape.Count; i++)
        {
            AddCandidate(new[] { i }, new[] { Lane });
            if (Rank > 1)
            {
                for (int j = i + 1; j < input.CheckedShape.Count; j++)
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
      IsWildcard("input", e => e is not Call { Target: IR.CPU.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
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
            var slice = IR.F.Tensors.Slice(packed, packedBegins, packedEnds, axes, strides);
            var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(slice, lanes, packAxes), candidate.CheckedShape.ToValueArray(), pads);
            if (post.CheckedType is not InvalidType)
            {
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
