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
        var inShape = input.CheckedShape;

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packedInput = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

            var resized = IR.F.CPU.ResizeImage(packedInput, packedAxes, padsInput.Select(x => (int)x.FixedValue).ToArray(), newSize, op.ResizeMode, op.TransformationMode, op.NearestMode);

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

public sealed class PackReduce : PackRule
{
    public PackReduce(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsReduce(
      "target",
      r => r.ReduceOp is ReduceOp.Mean or ReduceOp.Sum,
      IsWildcard("input", e => e is not Call { Target: IR.CPU.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
      IsTensorConst("axes") with { TypePattern = IsIntegral() },
      IsTensorConst("initValue") with { TypePattern = IsFloat() },
      IsTensorConst("keepDims") with { TypePattern = IsBool() });

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.Math.Reduce)result["target"];
        var input = (Expr)result["input"];
        var axes = ((TensorConst)result["axes"]).Value.ToArray<int>();
        if (axes.Length > 1)
        {
            return new();
        }

        var initValue = ((TensorConst)result["initValue"]).Value.ToScalar<float>();
        var keepDims = ((TensorConst)result["keepDims"]).Value.ToScalar<bool>();
        var inShape = input.CheckedShape;

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packedInput = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

            // todo support padings.
            if (padsInput.Any(x => !x.IsFixed || x.FixedValue != 0))
            {
                return;
            }

            Call reduce = IR.F.CPU.PackedReduce(packedInput, op.ReduceOp, axes, initValue, keepDims, packedAxes, padsInput.Select(x => (int)x.FixedValue).ToArray());

            var (outPackAxes, outPadNums, outLanes, outShape) = IR.CPU.PackedReduce.ComputeOutputInfo((IR.CPU.PackedReduce)reduce.Target, inShape, lanes);
            var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(reduce, outLanes, outPackAxes), outShape, outPadNums);

            if (post.CheckedType is not InvalidType)
            {
                rets.Add(post);
            }
        }

        for (int i = 0; i < input.CheckedShape.Count; i++)
        {
            AddCandidate([i], [Lane]);
            for (int j = i + 1; j < input.CheckedShape.Count; j++)
            {
                if (Rank > 1)
                {
                    AddCandidate([i, j], [Lane, Lane]);
                }
            }
        }

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
        var inShape = input.CheckedShape;
        var pshape = scale.CheckedShape;

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packedInput = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

            // todo support padings.
            if (padsInput.Any(x => !x.IsFixed || x.FixedValue != 0))
            {
                return;
            }

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

            var layernorm = IR.F.CPU.InstacneNorm(packedInput, packedScale, packedBias, eps, packedAxes, padsInput.Select(x => (int)x.FixedValue).ToArray());

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
    public PackMatMul(int rank = 2, int lane = 4, bool transB = false)
        : base(rank, lane)
    {
        TransB = transB;
    }

    public override Pattern Pattern { get; } = IsMatMul(
      "target",
      IsWildcard("lhs", e => e is not Call { Target: IR.CPU.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("rhs", e => e is not Call { Target: IR.CPU.Unpack }) with { TypePattern = IsFloat() & !IsVector() });

    /// <summary>
    /// Gets a value indicating whether trans b, note only for test.
    /// </summary>
    public bool TransB { get; }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var lhs = (Expr)result["lhs"];
        var rhs = (Expr)result["rhs"];
        var candidate = (Expr)result[Pattern];
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var rcontext = new RuleContext(rets, lhs, rhs, candidate, lhsShape, rhsShape);

        // pack A's k and B's k
        // AddCandidate(rcontext, PackKind.K, PackKind.K);

        // only pack A's m
        // AddCandidate(rcontext, PackKind.M, PackKind.None);

        // only pack B's n
        AddCandidate(rcontext, IR.CPU.PackedMatMul.PackKind.None, IR.CPU.PackedMatMul.PackKind.N/* , transB: rhs is Const */);
        if (Rank > 1)
        {
            // pack A's m and B's n, when B is const, force transpose
            AddCandidate(rcontext, IR.CPU.PackedMatMul.PackKind.M, IR.CPU.PackedMatMul.PackKind.N/* , transB: rhs is Const */);

            // pack A's m,k and B's k,n
            AddCandidate(rcontext, IR.CPU.PackedMatMul.PackKind.M | IR.CPU.PackedMatMul.PackKind.K, IR.CPU.PackedMatMul.PackKind.K | IR.CPU.PackedMatMul.PackKind.N/* , transB: rhs is Const */);
            if (TransB)
            {
                AddCandidate(rcontext, IR.CPU.PackedMatMul.PackKind.M | IR.CPU.PackedMatMul.PackKind.K, IR.CPU.PackedMatMul.PackKind.K | IR.CPU.PackedMatMul.PackKind.N, transB: TransB);
            }

            // pack A's m,k and B's k
            // AddCandidate(rcontext,  IR.CPU.PackedMatMul.PackKind.M |  IR.CPU.PackedMatMul.PackKind.K,  IR.CPU.PackedMatMul.PackKind.K);

            // pack A's k and B's k,n
            AddCandidate(rcontext, IR.CPU.PackedMatMul.PackKind.K, IR.CPU.PackedMatMul.PackKind.K | IR.CPU.PackedMatMul.PackKind.N/* , transB: lhs is Const */);
        }

        return rets;
    }

    private void AddCandidate(RuleContext context, IR.CPU.PackedMatMul.PackKind lhsPack, IR.CPU.PackedMatMul.PackKind rhsPack, bool transA = false, bool transB = false)
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
            case IR.CPU.PackedMatMul.PackKind.None:
                lhsLanes = Array.Empty<int>();
                lhsPackedAxes = Array.Empty<int>();
                break;
            case IR.CPU.PackedMatMul.PackKind.M:
                lhsLanes = [Lane];
                lhsPackedAxes = [lm];
                break;
            case IR.CPU.PackedMatMul.PackKind.K:
                lhsLanes = [Lane];
                lhsPackedAxes = [lk];
                break;
            case IR.CPU.PackedMatMul.PackKind.M | IR.CPU.PackedMatMul.PackKind.K:
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
            case IR.CPU.PackedMatMul.PackKind.None:
                rhsLanes = Array.Empty<int>();
                rhsPackedAxes = Array.Empty<int>();
                break;
            case IR.CPU.PackedMatMul.PackKind.N:
                rhsLanes = [Lane];
                rhsPackedAxes = [rn];
                break;
            case IR.CPU.PackedMatMul.PackKind.K:
                rhsLanes = [Lane];
                rhsPackedAxes = [rk];
                break;
            case IR.CPU.PackedMatMul.PackKind.K | IR.CPU.PackedMatMul.PackKind.N:
                rhsLanes = [Lane, Lane];
                rhsPackedAxes = [rk, rn];
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(rhsPack), rhsPack.ToString());
        }

        var packedLhs = IR.F.CPU.Pack(PackUtility.PadForPack(lhs, lhsShape, lhsPackedAxes, lhsLanes, 0f, out var lhsPadNums), lhsLanes, lhsPackedAxes);
        var packedRhs = IR.F.CPU.Pack(PackUtility.PadForPack(rhs, rhsShape, rhsPackedAxes, rhsLanes, 0f, out var rhsPadNums), rhsLanes, rhsPackedAxes);

        // TODO: support padding
        if (lhsPadNums.Any(x => !x.IsFixed || x.FixedValue > 0) || rhsPadNums.Any(x => !x.IsFixed || x.FixedValue > 0))
        {
            return;
        }

        var matmul = IR.F.CPU.PackedMatMul(packedLhs, packedRhs, lhsPackedAxes, lhsPadNums.Select(x => (int)x.FixedValue).ToArray(), rhsPackedAxes, rhsPadNums.Select(x => (int)x.FixedValue).ToArray(), transA, transB);

        var outRank = System.Math.Max(lhsShape.Length, rhsShape.Length);
        var lhsAlign = outRank - lhsShape.Length;
        var rhsAlign = outRank - rhsShape.Length;

        var unpackAxes = new List<int>();
        var unpadNums = new List<Dimension>();
        var unpackLanes = new List<int>();
        if (lhsPack.HasFlag(IR.CPU.PackedMatMul.PackKind.M))
        {
            var mPackIndex = Array.IndexOf(lhsPackedAxes, lm);
            unpackAxes.Add(outRank - 2);
            unpadNums.Add(lhsPadNums[mPackIndex]);
            unpackLanes.Add(Lane);
        }

        if (rhsPack.HasFlag(IR.CPU.PackedMatMul.PackKind.N))
        {
            var nPackIndex = Array.IndexOf(rhsPackedAxes, rn);
            unpackAxes.Add(outRank - 1);
            unpadNums.Add(rhsPadNums[nPackIndex]);
            unpackLanes.Add(Lane);
        }

        Expr post = matmul;
        if (unpackAxes.Any())
        {
            post = PackUtility.SliceForPack(IR.F.CPU.Unpack(matmul, unpackLanes.ToArray(), unpackAxes.ToArray()), candidate.CheckedShape, unpadNums.ToArray());
        }

        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }
    }

    private sealed record RuleContext(List<Expr> Results, Expr Lhs, Expr Rhs, Expr Candidate, Shape LhsShape, Shape RhsShape)
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
        var inShape = input.CheckedShape;

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packedInput = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

            // todo support padings.
            if (padsInput.Any(x => !x.IsFixed))
            {
                return;
            }

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
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;

        void AddCandidate(int[] lhsPackedAxes, int[] rhsPackedAxes, int[] lhsLanes, int[] rhsLanes)
        {
            var packedLhs = IR.F.CPU.Pack(PackUtility.PadForPack(lhs, lhsShape, lhsPackedAxes, lhsLanes, 0f, out var lhsPadNums), lhsLanes, lhsPackedAxes);
            var packedRhs = IR.F.CPU.Pack(PackUtility.PadForPack(rhs, rhsShape, rhsPackedAxes, rhsLanes, 0f, out var rhsPadNums), rhsLanes, rhsPackedAxes);

            // todo support padings.
            if (lhsPadNums.Any(x => !x.IsFixed)
                || rhsPadNums.Any(x => !x.IsFixed))
            {
                return;
            }

            var binary = IR.F.CPU.PackedBinary(packedLhs, packedRhs, op.BinaryOp, lhsPackedAxes, lhsPadNums.Select(x => (int)x.FixedValue).ToArray(), rhsPackedAxes, rhsPadNums.Select(x => (int)x.FixedValue).ToArray());
            var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(binary, lhsLanes.Length >= rhsLanes.Length ? lhsLanes : rhsLanes, lhsPackedAxes.Length >= rhsPackedAxes.Length ? lhsPackedAxes : rhsPackedAxes), candidate.CheckedShape, lhsPackedAxes.Length >= rhsPackedAxes.Length ? lhsPadNums : rhsPadNums);
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

    public IEnumerable<int[]> GeneratePackAxes(Shape shape)
    {
        if (shape.IsUnranked || shape.Rank == 0 || (shape.Rank == 1 && shape[0].IsFixed && shape[0].FixedValue == 1))
        {
            yield return Array.Empty<int>();
        }
        else
        {
            for (int i = 0; i < shape.Rank; i++)
            {
                if (shape[i].IsFixed)
                {
                    yield return new[] { i };
                    for (int j = i + 1; j < shape.Rank; j++)
                    {
                        if (shape[j].IsFixed)
                        {
                            yield return new[] { i, j };
                        }
                    }
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
        var inShape = input.CheckedShape;

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packed = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var pads), lanes, packedAxes);

            // todo support padings.
            if (pads.Any(x => !x.IsFixed))
            {
                return;
            }

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
    public PackTranspose(int rank = 2, int lane = 4)
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
        var inShape = input.CheckedShape;

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packed = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var pads), lanes, packedAxes);

            // todo support padings.
            if (pads.Any(x => !x.IsFixed))
            {
                return;
            }

            var tarns = IR.F.Tensors.Transpose(packed, perm);
            if (tarns.CheckedType is not InvalidType)
            {
                var partialPerm = perm.Select(axis => packedAxes.IndexOf(axis)).Where(x => x != -1).ToArray();
                var unpackAxes = packedAxes.Select(axis => perm.IndexOf(axis)).ToArray();
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
        var inShape = input.CheckedShape;

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packed = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var pads), lanes, packedAxes);

            // todo support padings.
            if (pads.Any(x => !x.IsFixed))
            {
                return;
            }

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

    public static Expr AddCandidate(Expr input, Expr weights, Expr bias, int[] strides, int[] padding, long[] wShape, long[] outShape)
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

    public static Expr AddPackedCandidate(Expr input, Expr weights, Expr bias, int[] strides, int[] padding, long[] wShape, long[] outShape, int lane)
    {
        var paddedInput = PackUtility.PadForPack(input, input.CheckedShape.ToValueArray(), new[] { 1 }, new[] { lane }, 0f, out _);
        var col = IR.F.CPU.Im2col(IR.F.CPU.Pack(paddedInput, new[] { lane }, new[] { 1 }), new[] { wShape[2], wShape[3] }, strides, padding, new[] { 1 }, new[] { 0 });
        var paddedW = PackUtility.PadForPack(weights, wShape, new[] { 1 }, new[] { lane }, 0f, out _);
        var newW = IR.F.Tensors.Reshape(IR.F.CPU.Pack(paddedW, new[] { lane }, new[] { 1 }), new[] { wShape[0], MathUtility.CeilDiv(wShape[1], lane) * wShape[2] * wShape[3] });
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
      IsWildcard("input", e => e is not Call { Target: IR.CPU.Unpack }) with { TypePattern = !IsVector() & HasFixedShape() },
      IsTensorConst("newShape") with { TypePattern = IsIntegral() });

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

        var input = (Expr)result["input"];
        var newShape = ((TensorConst)result["newShape"]).Value.ToArray<long>();
        var inShape = input.CheckedShape.ToValueArray();

        // 1. find the mapping transforms
        if (!IRUtility.TryGetShapeMapMatrix(inShape, newShape, out var mat))
        {
            return new List<Expr> { };
        }

        var (forwardDict, backwardDict) = IRUtility.ShapeMapMatrixAsDict(mat);

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

            if (unpackAxes.Count == 0)
            {
                return;
            }

            var packed = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var pads), lanes, packedAxes);

            // todo support padings.
            if (pads.Any(x => !x.IsFixed))
            {
                return;
            }

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
        var inShape = input.CheckedShape;
        var candidate = (Expr)result[Pattern];
        for (int i = 0; i < axes.Length; i++)
        {
            ends[i] = ends[i] switch
            {
                < 0 => inShape[axes[i]].FixedValue + ends[i],
                int.MaxValue => inShape[axes[i]].FixedValue,
                long.MaxValue => inShape[axes[i]].FixedValue,
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

            // todo support padings.
            if (pads.Any(x => !x.IsFixed))
            {
                return;
            }

            var slice = IR.F.Tensors.Slice(packed, packedBegins, packedEnds, axes, strides);
            var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(slice, lanes, packAxes), candidate.CheckedShape, pads);
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

public sealed class PackCast : PackRule
{
    public PackCast(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsCast(
      "target",
      _ => true,
      IsWildcard("input", e => e is not Call { Target: IR.CPU.Unpack }) with { TypePattern = IsFloat() & !IsVector() });

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.Tensors.Cast)result["target"];
        var input = (Expr)result["input"];
        var inShape = input.CheckedShape;

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packedInput = IR.F.CPU.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

            // todo support padings.
            if (padsInput.Any(x => !x.IsFixed))
            {
                return;
            }

            var cast = IR.F.Tensors.Cast(packedInput, op.NewType, op.CastMode);
            var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(cast, lanes, packedAxes), inShape, padsInput);
            if (cast.CheckedType is not InvalidType)
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

[RuleGenerator]
public sealed partial class TransposePackMatMulInputs : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = PatternMatch.F.CPU.IsPackedMatMul("matmul", "caller", m => m.RhsPackedAxes.Count == 2 && m.RhsPadedNums.All(v => v == 0) && m.TransposeB == false, IsWildcard("lhs"), PatternMatch.F.CPU.IsPack("rhsPack", "callee", p => p.Axes.Count == 2 && p.Lanes.Count == 2, PatternMatch.F.Tensors.IsTranspose("trans", "rhs", IsWildcard("transInput"), IsTensorConst("perm") /* IsAlt(IsTensorConst("rhs"), PatternMatch.F.Tensors.IsTranspose("trans", "rhs", IsWildcard("transInput"), IsTensorConst("perm")) */)));

    private Expr? GetReplace(IR.CPU.PackedMatMul matmul, Expr lhs, IR.CPU.Pack rhsPack, Expr transInput, int[] perm, IMatchResult result)
    {
        // note can't enable transpose const b, because const folding will be very solw.
        var inputsShape = transInput.CheckedShape.ToValueArray();
        var tperm = Enumerable.Range(0, inputsShape.Length).ToArray();
        (tperm[^2], tperm[^1]) = (tperm[^1], tperm[^2]);
        if (tperm.SequenceEqual(perm))
        {
            var npack = IR.F.CPU.Pack(transInput, [rhsPack.Lanes[1], rhsPack.Lanes[0]], [rhsPack.Axes[1], rhsPack.Axes[0]]);
            var newMatmul = new IR.CPU.PackedMatMul(matmul.LhsPackedAxes, matmul.LhsPadedNums, new[] { matmul.RhsPackedAxes[1], matmul.RhsPackedAxes[0] }, new[] { matmul.RhsPadedNums[1], matmul.RhsPadedNums[0] }, false, true, matmul.FusedReduce);
            return new Call(newMatmul, lhs, npack);
        }

        return null;
    }
}
