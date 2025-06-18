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
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Imaging;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

public abstract class PackRule : RewriteRule<Pattern>
{
    public PackRule(int rank, int lane)
    {
        Rank = rank;
        Lane = lane;
    }

    public int Lane { get; }

    public int Rank { get; }

    public override BaseExpr? GetReplace(IMatchResult result, RunPassContext options) => throw new NotImplementedException();

    public IEnumerable<int[]> GeneratePackAxes(Shape shape)
    {
        if (shape.IsUnranked || shape.Rank == 0 || (shape.Rank == 1 && shape[0].IsFixed && shape[0].FixedValue == 1))
        {
            yield return Array.Empty<int>();
        }
        else
        {
            yield return Array.Empty<int>();
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

    public (int InitPad, int ExtraPad) FindMinimumPad(int s, int lane, int hierarchy)
    {
        var initPad = (lane - (s % lane)) % lane;
        var pad = 0;

        while ((s + initPad + pad) / lane % hierarchy != 0)
        {
            pad += lane;
        }

        return (initPad, pad);
    }
}

public sealed class PackResizeImage : PackRule
{
    public PackResizeImage(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsResizeImage("target", op => op.TransformationMode == ImageResizeTransformationMode.Asymmetric && op.IsTFResize == false, IsWildcard("input") with { TypePattern = !IsVector() }, IsWildcard("roi"), IsFixedShape("newSize"), IsTensorConst("cubicCoeffA"), IsTensorConst("excludeOutside"), IsTensorConst("extrapolationValue"));

    public static List<Expr> AddCandidate(IR.Imaging.ResizeImage op, Expr input, int[] newSize, int[] packedAxes, int[] lanes)
    {
        var inShape = input.CheckedShape;

        var rets = new List<Expr>();
        if (packedAxes.Length == 0)
        {
            return rets;
        }

        var packedInput = IR.F.Tensors.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

        var resized = IR.F.NTT.ResizeImage(packedInput, new RankedShape(padsInput), packedAxes, newSize, op.ResizeMode, op.TransformationMode, op.NearestMode);

        var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(resized, lanes, packedAxes), inShape, padsInput!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var roi = (Expr)result["roi"];
        if (roi is not None && ((RankedShape)roi.CheckedShape).Size != 0)
        {
            return rets;
        }

        var op = (IR.Imaging.ResizeImage)result["target"];
        var input = (Expr)result["input"];
        var newSize = ((RankedShape)result["newSize"]).ToValueArray().ToInts();
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        rets = AddCandidate(op, input, newSize, [1], [laneSize]);
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
      IsWildcard("input", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = IsFloat() & !IsVector() & HasRankedShape() },
      IsFixedShape("axes"),
      IsTensorConst("initValue") with { TypePattern = IsFloat() },
      IsTensorConst("keepDims") with { TypePattern = IsBool() });

    public static List<Expr> AddCandidate(IR.Math.Reduce op, Expr input, int[] axes, float initValue, bool keepDims, int[] packedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var inShape = (RankedShape)input.CheckedShape;
        if (packedAxes.Length == 0)
        {
            return rets;
        }

        axes = axes.Select(x => (int)Util.PositiveIndex(x, inShape.Rank)).ToArray();
        var packedInput = IR.F.Tensors.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

        Call reduce = IR.F.NTT.PackedReduce(packedInput, op.ReduceOp, axes, initValue, keepDims, packedAxes, new RankedShape(padsInput));

        var (outPackAxes, outPadNums, outLanes, outShape) = IR.NTT.PackedReduce.ComputeOutputInfo((IR.NTT.PackedReduce)reduce.Target, padsInput, inShape, lanes);
        var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(reduce, outLanes, outPackAxes), outShape, outPadNums);

        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override IReadOnlyList<BaseExpr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.Math.Reduce)result["target"];
        var input = (Expr)result["input"];
        var axes = ((RankedShape)result["axes"]).ToValueArray().ToInts();
        if (axes.Length > 1)
        {
            return Array.Empty<BaseExpr>();
        }

        var initValue = ((TensorConst)result["initValue"]).Value.ToScalar<float>();
        var keepDims = ((TensorConst)result["keepDims"]).Value.ToScalar<bool>();
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;
        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(op, input, axes, initValue, keepDims, [i], [laneSize]));
            for (int j = i + 1; j < input.CheckedShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    rets.AddRange(AddCandidate(op, input, axes, initValue, keepDims, [i, j], [laneSize, laneSize]));
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

    public override IReadOnlyList<BaseExpr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.NN.InstanceNormalization)result["target"];
        var input = (Expr)result["input"];
        var scale = (Expr)result["scale"];
        var bias = (Expr)result["bias"];
        var eps = ((TensorConst)result["eps"]).Value.ToScalar<float>();
        var inShape = input.CheckedShape;
        var pshape = scale.CheckedShape;
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packedInput = IR.F.Tensors.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

            var pAxes = packedAxes.Where(i => i == 1).Select(i => 0).ToArray();
            var packedScale = PackUtility.PadForPack(scale, pshape, pAxes, lanes, 0f, out var padsScale);
            if (pAxes.Length > 0)
            {
                packedScale = IR.F.Tensors.Pack(packedScale, Enumerable.Repeat(laneSize, pAxes.Length).ToArray(), pAxes);
            }

            var packedBias = PackUtility.PadForPack(bias, pshape, pAxes, lanes, 0f, out var padsBias);
            if (pAxes.Length > 0)
            {
                packedBias = IR.F.Tensors.Pack(packedBias, Enumerable.Repeat(laneSize, pAxes.Length).ToArray(), pAxes);
            }

            var instanceNorm = IR.F.NTT.InstacneNorm(packedInput, packedScale, packedBias, eps, packedAxes, new RankedShape(padsInput));

            if (instanceNorm.CheckedType is not InvalidType)
            {
                var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(instanceNorm, lanes, packedAxes), inShape, padsInput);
                rets.Add(post);
            }
        }

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            AddCandidate(new[] { i }, new[] { laneSize });
        }

        return rets;
    }
}

public sealed class PackMatMul : PackRule
{
    public PackMatMul(int rank = 2, int lane = 16, bool transB = false)
        : base(rank, lane)
    {
        TransB = transB;
    }

    public override Pattern Pattern { get; } = IsMatMul(
      "matmul",
      "target",
      (dytpe) => true,
      IsWildcard("lhs", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("rhs", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = IsFloat() & !IsVector() });

    /// <summary>
    /// Gets a value indicating whether trans b, note only for test.
    /// </summary>
    public bool TransB { get; }

    public override IReadOnlyList<BaseExpr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var lhs = (Expr)result["lhs"];
        var rhs = (Expr)result["rhs"];
        var candidate = (Expr)result[Pattern];
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var outputDataType = ((Nncase.IR.Math.MatMul)result["matmul"]).OutputDataType;
        var rcontext = new RuleContext(rets, lhs, rhs, candidate, lhsShape, rhsShape, outputDataType);

        // pack A's k and B's k
        // AddCandidate(rcontext, PackKind.K, PackKind.K);

        // only pack A's m
        // AddCandidate(rcontext, PackKind.M, PackKind.None);

        // only pack B's n
        AddCandidate(rcontext, IR.NTT.PackedMatMul.PackKind.None, IR.NTT.PackedMatMul.PackKind.N/* , transB: rhs is Const */);
        if (Rank > 1)
        {
            // pack A's m and B's n, when B is const, force transpose
            AddCandidate(rcontext, IR.NTT.PackedMatMul.PackKind.M, IR.NTT.PackedMatMul.PackKind.N/* , transB: rhs is Const */);

            // pack A's m,k and B's k,n
            AddCandidate(rcontext, IR.NTT.PackedMatMul.PackKind.M | IR.NTT.PackedMatMul.PackKind.K, IR.NTT.PackedMatMul.PackKind.K | IR.NTT.PackedMatMul.PackKind.N/* , transB: rhs is Const */);
            if (TransB)
            {
                AddCandidate(rcontext, IR.NTT.PackedMatMul.PackKind.M | IR.NTT.PackedMatMul.PackKind.K, IR.NTT.PackedMatMul.PackKind.K | IR.NTT.PackedMatMul.PackKind.N, transB: TransB);
            }

            // pack A's m,k and B's k
            // AddCandidate(rcontext,  IR.NTT.PackedMatMul.PackKind.M |  IR.NTT.PackedMatMul.PackKind.K,  IR.NTT.PackedMatMul.PackKind.K);

            // pack A's k and B's k,n
            AddCandidate(rcontext, IR.NTT.PackedMatMul.PackKind.K, IR.NTT.PackedMatMul.PackKind.K | IR.NTT.PackedMatMul.PackKind.N/* , transB: lhs is Const */);
        }

        return rets;
    }

    private void AddCandidate(RuleContext context, IR.NTT.PackedMatMul.PackKind lhsPack, IR.NTT.PackedMatMul.PackKind rhsPack, bool transA = false, bool transB = false)
    {
        var (rets, lhs, rhs, candidate, _, _, outputDataType) = context;
        var lhsShape = context.LhsShape.ToArray();
        var rhsShape = context.RhsShape.ToArray();
        var lhsLaneSize = Lane / lhs.CheckedDataType.SizeInBytes;
        var rhsLaneSize = Lane / rhs.CheckedDataType.SizeInBytes;
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
            case IR.NTT.PackedMatMul.PackKind.None:
                lhsLanes = Array.Empty<int>();
                lhsPackedAxes = Array.Empty<int>();
                break;
            case IR.NTT.PackedMatMul.PackKind.M:
                lhsLanes = [lhsLaneSize];
                lhsPackedAxes = [lm];
                break;
            case IR.NTT.PackedMatMul.PackKind.K:
                lhsLanes = [lhsLaneSize];
                lhsPackedAxes = [lk];
                break;
            case IR.NTT.PackedMatMul.PackKind.M | IR.NTT.PackedMatMul.PackKind.K:
                lhsLanes = [lhsLaneSize, lhsLaneSize];
                lhsPackedAxes = [lm, lk];
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(lhsPack), lhsPack.ToString());
        }

        int[] rhsLanes;
        int[] rhsPackedAxes;
        switch (rhsPack)
        {
            case IR.NTT.PackedMatMul.PackKind.None:
                rhsLanes = Array.Empty<int>();
                rhsPackedAxes = Array.Empty<int>();
                break;
            case IR.NTT.PackedMatMul.PackKind.N:
                rhsLanes = [rhsLaneSize];
                rhsPackedAxes = [rn];
                break;
            case IR.NTT.PackedMatMul.PackKind.K:
                rhsLanes = [rhsLaneSize];
                rhsPackedAxes = [rk];
                break;
            case IR.NTT.PackedMatMul.PackKind.K | IR.NTT.PackedMatMul.PackKind.N:
                rhsLanes = [rhsLaneSize, rhsLaneSize];
                rhsPackedAxes = [rk, rn];
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(rhsPack), rhsPack.ToString());
        }

        var packedLhs = IR.F.Tensors.Pack(PackUtility.PadForPack(lhs, lhsShape, lhsPackedAxes, lhsLanes, 0f, out var lhsPadNums), lhsLanes, lhsPackedAxes);
        var packedRhs = IR.F.Tensors.Pack(PackUtility.PadForPack(rhs, rhsShape, rhsPackedAxes, rhsLanes, 0f, out var rhsPadNums), rhsLanes, rhsPackedAxes);

        var matmul = IR.F.NTT.PackedMatMul(packedLhs, packedRhs, lhsPackedAxes, rhsPackedAxes, transA, transB, false, outputDataType);

        var outRank = System.Math.Max(lhsShape.Length, rhsShape.Length);
        _ = outRank - lhsShape.Length;
        _ = outRank - rhsShape.Length;

        var unpackAxes = new List<int>();
        var unpadNums = new List<Dimension>();
        var unpackLanes = new List<int>();
        if (lhsPack.HasFlag(IR.NTT.PackedMatMul.PackKind.M))
        {
            var mPackIndex = Array.IndexOf(lhsPackedAxes, lm);
            unpackAxes.Add(outRank - 2);
            unpadNums.Add(lhsPadNums[mPackIndex]);
            unpackLanes.Add(lhsLaneSize);
        }

        if (rhsPack.HasFlag(IR.NTT.PackedMatMul.PackKind.N))
        {
            var nPackIndex = Array.IndexOf(rhsPackedAxes, rn);
            unpackAxes.Add(outRank - 1);
            unpadNums.Add(rhsPadNums[nPackIndex]);
            unpackLanes.Add(rhsLaneSize);
        }

        Expr post = matmul;
        if (unpackAxes.Any())
        {
            post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(matmul, unpackLanes.ToArray(), unpackAxes.ToArray()), candidate.CheckedShape, unpadNums.ToArray());
        }

        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }
    }

    private sealed record RuleContext(List<Expr> Results, Expr Lhs, Expr Rhs, Expr Candidate, Shape LhsShape, Shape RhsShape, DataType OutputDataType)
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
      IsWildcard("input", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = IsFloat() & !IsVector() });

    public static List<Expr> AddCandidate(IR.Math.Unary op, Expr input, int[] packedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var inShape = input.CheckedShape;
        var packedInput = IR.F.Tensors.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

        var unary = IR.F.Math.Unary(op.UnaryOp, packedInput);
        var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(unary, lanes, packedAxes), inShape, padsInput);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.Math.Unary)result["target"];
        var input = (Expr)result["input"];
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(op, input, [i], [laneSize]));
            for (int j = i + 1; j < input.CheckedShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    rets.AddRange(AddCandidate(op, input, [i, j], [laneSize, laneSize]));
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
      IsWildcard("lhs", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("rhs", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = IsFloat() & !IsVector() });

    public static List<Expr> AddCandidate(IR.Math.Binary op, Expr lhs, Expr rhs, Expr candidate, int[] lhsPackedAxes, int[] rhsPackedAxes, int[] lhsLanes, int[] rhsLanes)
    {
        var rets = new List<Expr>();
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var outShape = candidate.CheckedShape;
        if (lhsPackedAxes.Length == 0 && rhsPackedAxes.Length == 0)
        {
            return rets;
        }

        var alignedLhsPackedAxes = lhsPackedAxes.Select(a => a + outShape.Rank - lhsShape.Rank).ToArray();
        var alignedRhsPackedAxes = rhsPackedAxes.Select(a => a + outShape.Rank - rhsShape.Rank).ToArray();
        var alignedLhsShape = Enumerable.Repeat(new DimConst(1), outShape.Rank - lhsShape.Rank).Concat(lhsShape).ToArray();
        var alignedRhsShape = Enumerable.Repeat(new DimConst(1), outShape.Rank - rhsShape.Rank).Concat(rhsShape).ToArray();
        if (alignedLhsPackedAxes.Any(a => alignedRhsShape[a] is { IsFixed: true, FixedValue: var d } && d != 1 && !alignedRhsPackedAxes.Contains(a))
        || alignedRhsPackedAxes.Any(a => alignedLhsShape[a] is { IsFixed: true, FixedValue: var d } && d != 1 && !alignedLhsPackedAxes.Contains(a)))
        {
            return rets;
        }

        var packedLhs = IR.F.Tensors.Pack(PackUtility.PadForPack(lhs, lhsShape, lhsPackedAxes, lhsLanes, 0f, out var lhsPadNums), lhsLanes, lhsPackedAxes);
        var packedRhs = IR.F.Tensors.Pack(PackUtility.PadForPack(rhs, rhsShape, rhsPackedAxes, rhsLanes, 0f, out var rhsPadNums), rhsLanes, rhsPackedAxes);

        var binary = IR.F.NTT.PackedBinary(packedLhs, packedRhs, op.BinaryOp, lhsPackedAxes, lhsPadNums, rhsPackedAxes, rhsPadNums);
        var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(binary, lhsLanes.Length >= rhsLanes.Length ? lhsLanes : rhsLanes, lhsPackedAxes.Length >= rhsPackedAxes.Length ? alignedLhsPackedAxes : alignedRhsPackedAxes), candidate.CheckedShape, lhsPackedAxes.Length >= rhsPackedAxes.Length ? lhsPadNums! : rhsPadNums!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.Math.Binary)result["target"];
        var lhs = (Expr)result["lhs"];
        var rhs = (Expr)result["rhs"];
        var candidate = (Expr)result[Pattern];
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var lhsLaneSize = Lane / lhs.CheckedDataType.SizeInBytes;
        var rhsLaneSize = Lane / rhs.CheckedDataType.SizeInBytes;

        foreach (var arr in new[] { GeneratePackAxes(lhsShape), GeneratePackAxes(rhsShape) }.CartesianProduct())
        {
            var lhsPackedAxes = arr.First();
            var rhsPackedAxes = arr.Skip(1).First();
            if (lhsPackedAxes.Length <= Rank && rhsPackedAxes.Length <= Rank)
            {
                rets.AddRange(AddCandidate(op, lhs, rhs, candidate, lhsPackedAxes, rhsPackedAxes, Enumerable.Repeat(lhsLaneSize, lhsPackedAxes.Length).ToArray(), Enumerable.Repeat(rhsLaneSize, rhsPackedAxes.Length).ToArray()));
            }
        }

        return rets;
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
      IsWildcard("input", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
      IsTensorConst("beta") with { TypePattern = IsFloatScalar() });

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var input = (Expr)result["input"];
        var beta = ((TensorConst)result["beta"]).Value.ToScalar<float>();
        var inShape = input.CheckedShape;
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        void AddCandidate(int[] packedAxes, int[] lanes)
        {
            var packed = IR.F.Tensors.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var pads), lanes, packedAxes);

            var swish = IR.F.NN.Swish(packed, beta);
            var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(swish, lanes, packedAxes), inShape, pads);
            if (post.CheckedType is not InvalidType)
            {
                rets.Add(post);
            }
        }

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            AddCandidate(new[] { i }, new[] { laneSize });
            for (int j = i + 1; j < input.CheckedShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    AddCandidate(new[] { i, j }, new[] { laneSize, laneSize });
                }
            }
        }

        return rets;
    }
}

public sealed class PackTranspose : PackRule
{
    public PackTranspose(int rank = 2, int lane = 16)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsTranspose(
      "trans",
      IsWildcard("input", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
      IsFixedShape("perm"));

    public static List<Expr> AddCandidate(Expr input, int[] perm, int[] packedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var inShape = input.CheckedShape;
        var packed = IR.F.Tensors.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var pads), lanes, packedAxes);

        var tarns = IR.F.Tensors.Transpose(packed, perm);
        if (tarns.CheckedType is not InvalidType)
        {
            var partialPerm = perm.Select(axis => packedAxes.IndexOf(axis)).Where(x => x != -1).ToArray();
            var unpackAxes = packedAxes.Select(axis => perm.IndexOf(axis)).ToArray();
            var unpackPads = Enumerable.Range(0, pads.Length).Select(i => pads[partialPerm[i]]).ToArray();
            var unpackLanes = Enumerable.Range(0, lanes.Length).Select(i => lanes[partialPerm[i]]).ToArray();
            var newShape = perm.Select(i => inShape[i]).ToArray();
            var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(tarns, unpackLanes, unpackAxes), newShape, unpackPads);
            if (post.CheckedType is not InvalidType)
            {
                rets.Add(post);
            }
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

        var input = (Expr)result["input"];
        var perm = ((RankedShape)result["perm"]).ToValueArray().ToInts();
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(input, perm, [i], [laneSize]));
            for (int j = i + 1; j < input.CheckedShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    rets.AddRange(AddCandidate(input, perm, [i, j], [laneSize, laneSize]));
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
      IsWildcard("input", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
      IsFixedShape("axes"));

    public static List<Expr> AddCandidate(Expr input, int[] axes, int[] packedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var inShape = input.CheckedShape;
        if (packedAxes.Length == 0)
        {
            return rets;
        }

        var packed = IR.F.Tensors.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var pads), lanes, packedAxes);

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

        var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(unseq, lanes, unpackAxes), outShape.ToArray(), pads!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

        var input = (Expr)result["input"];
        var axes = ((RankedShape)result["axes"]).ToValueArray().ToInts();
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(input, axes, [i], [laneSize]));
            for (int j = i + 1; j < input.CheckedShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    rets.AddRange(AddCandidate(input, axes, [i, j], [laneSize, laneSize]));
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
        IsWildcard("input", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = !IsVector() },
        IsWildcard("weights"),
        IsWildcard("bias"),
        IsFixedShape("stride"),
        IsFixedPaddings("padding"),
        IsFixedShape("dilation"),
        IsFixedDimension("groups"),
        IsTensorConst("fusedClamp"));

    public static Expr AddCandidate(Expr input, Expr weights, Expr bias, int[] strides, int[] padding, long[] wShape, long[] outShape)
    {
        var col = IR.F.NTT.Im2col(input, new[] { wShape[2], wShape[3] }, strides, padding);
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
        var col = IR.F.NTT.Im2col(IR.F.Tensors.Pack(paddedInput, new[] { lane }, new[] { 1 }), new[] { wShape[2], wShape[3] }, strides, padding, new[] { 1 }, new[] { 0 });
        var paddedW = PackUtility.PadForPack(weights, wShape, new[] { 1 }, new[] { lane }, 0f, out _);
        var newW = IR.F.Tensors.Reshape(IR.F.Tensors.Pack(paddedW, new[] { lane }, new[] { 1 }), new[] { wShape[0], MathUtility.CeilDiv(wShape[1], lane) * wShape[2] * wShape[3] });
        var matmul = IR.F.NTT.PackedMatMul(newW, col, new[] { 1 }, new[] { 0 }, false, false, false); // [oc, b*oh*ow]
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

    public override IReadOnlyList<BaseExpr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

        var input = (Expr)result["input"];
        var weights = (Expr)result["weights"];
        var bias = (Expr)result["bias"];
        var strides = ((RankedShape)result["stride"]).ToValueArray().ToInts();
        var padding = ((RankedShape)result["padding"]).ToValueArray().ToInts();
        var dilation = ((RankedShape)result["dilation"]).ToValueArray().ToInts();
        var groups = (int)((DimConst)result["groups"]).Value;
        var fusedClamp = ((TensorConst)result["fusedClamp"]).Value.ToArray<float>();
        var wShape = weights.CheckedShape.ToValueArray();
        var outShape = ((Expr)result[Pattern]).CheckedShape.ToValueArray();
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;
        if (groups != 1 || wShape[1] % Lane != 0 || dilation[0] != 1 || dilation[1] != 1 || fusedClamp[0] != float.NegativeInfinity || fusedClamp[1] != float.PositiveInfinity)
        {
            return rets;
        }

        // only pack on in channels
        rets.Add(AddCandidate(input, weights, bias, strides, padding, wShape, outShape));
        rets.Add(AddPackedCandidate(input, weights, bias, strides, padding, wShape, outShape, laneSize));
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
      IsWildcard("input", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = !IsVector() & HasFixedShape() },
      IsFixedShape("newShape"));

    public static List<Expr> AddCandidate(Expr input, long[] newShape, Dictionary<int, List<int>> forwardDict, Dictionary<int, List<int>> backwardDict, int[] packedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var inShape = input.CheckedShape;
        if (packedAxes.Length == 0)
        {
            return rets;
        }

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
                if (inAxes.Count == 1 || (inAxes[^1] == axis && inShape[axis] % lanes[packedAxes.IndexOf(axis)] == 0))
                {
                    unpackAxes.Add(outAxis);
                }
                else
                {
                    return rets;
                }
            }
        }

        if (unpackAxes.Count == 0 || unpackAxes.Count != packedAxes.Length)
        {
            return rets;
        }

        var packed = IR.F.Tensors.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var pads), lanes, packedAxes);

        var packedNewShape = newShape.ToArray();
        foreach (var (lane, axis) in lanes.Zip(unpackAxes))
        {
            packedNewShape[axis] = MathUtility.CeilDiv(packedNewShape[axis], lane);
        }

        var nreshape = IR.F.Tensors.Reshape(packed, packedNewShape);
        var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(nreshape, lanes, unpackAxes.ToArray()), newShape, pads!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

        var input = (Expr)result["input"];
        var newShape = ((RankedShape)result["newShape"]).ToValueArray();
        var inShape = input.CheckedShape.ToValueArray();
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        // 1. find the mapping transforms
        if (!IRUtility.TryGetShapeMapMatrix(inShape, newShape, out var mat))
        {
            return new List<Expr> { };
        }

        var (forwardDict, backwardDict) = IRUtility.ShapeMapMatrixAsDict(mat);

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(input, newShape, forwardDict, backwardDict, new[] { i }, new[] { laneSize }));
            if (Rank > 1)
            {
                for (int j = i + 1; j < input.CheckedShape.Rank; j++)
                {
                    rets.AddRange(AddCandidate(input, newShape, forwardDict, backwardDict, new[] { i, j }, new[] { laneSize, laneSize }));
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
      IsWildcard("input", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
      IsFixedShape("begins"),
      IsFixedShape("ends"),
      IsFixedShape("axes"),
      IsFixedShape("strides"));

    public static List<Expr> AddCandidate(Expr input, Expr candidate, long[] begins, long[] ends, long[] axes, long[] strides, int[] packAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        if (packAxes.Length == 0)
        {
            return rets;
        }

        var inShape = input.CheckedShape;
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
                    return rets;
                }
            }
        }

        var packed = IR.F.Tensors.Pack(PackUtility.PadForPack(input, inShape, packAxes, lanes, 0f, out var pads), lanes, packAxes);

        var slice = IR.F.Tensors.Slice(packed, packedBegins, packedEnds, axes, strides);
        var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(slice, lanes, packAxes), candidate.CheckedShape, pads!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

        var input = (Expr)result["input"];
        var begins = ((RankedShape)result["begins"]).ToValueArray();
        var ends = ((RankedShape)result["ends"]).ToValueArray();
        var axes = ((RankedShape)result["axes"]).ToValueArray();
        var strides = ((RankedShape)result["strides"]).ToValueArray();
        var inShape = input.CheckedShape;
        var candidate = (Expr)result[Pattern];
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;
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

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(input, candidate, begins, ends, axes, strides, [i], [laneSize]));
            for (int j = i + 1; j < input.CheckedShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    rets.AddRange(AddCandidate(input, candidate, begins, ends, axes, strides, new[] { i, j }, new[] { laneSize, laneSize }));
                }
            }
        }

        return rets;
    }
}

public sealed class PackConcat : PackRule
{
    public PackConcat(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsConcat(
        "concat",
        "concatCall",
        _ => true,
        IsTuple(null, IsVArgsRepeat("tupleInputs", exprs =>
        {
            var patterns = new Pattern[exprs.Length];
            for (var i = 0; i < patterns.Length; i++)
            {
                patterns[i] = IsWildcard($"input_{i}", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = IsFloat() & !IsVector() };
            }

            return patterns;
        })));

    public static List<Expr> AddCandidate(BaseExpr[] inputs, Expr candidate, int axis, int[] packAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        if (packAxes.Length == 0)
        {
            return rets;
        }

        var packedInputs = new Expr[inputs.Length];
        Dimension[]? pads = null;
        for (var i = 0; i < inputs.Length; i++)
        {
            var inShape = inputs[i].CheckedShape;
            packedInputs[i] = IR.F.Tensors.Pack(PackUtility.PadForPack((Expr)inputs[i], inShape, packAxes, lanes, 0f, out pads), lanes, packAxes);
        }

        var concat = IR.F.Tensors.Concat(new IR.Tuple(packedInputs), axis);
        var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(concat, lanes, packAxes), candidate.CheckedShape, pads!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

        var tupleInputs = (IReadOnlyList<BaseExpr>)result["tupleInputs"];
        var op = (IR.Tensors.Concat)result["concat"];
        var axis = op.Axis;
        var candidate = (Expr)result[Pattern];
        var laneSize = Lane / tupleInputs[0].CheckedDataType.SizeInBytes;

        for (int i = 0; i < tupleInputs[0].CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(tupleInputs.ToArray(), candidate, axis, [i], [laneSize]));
        }

        return rets;
    }
}

public sealed class PackCompare : PackRule
{
    public PackCompare(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsCompare(
      "target",
      _ => true,
      IsWildcard("lhs", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("rhs", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = IsFloat() & !IsVector() });

    public static List<Expr> AddCandidate(IR.Math.Compare op, Expr lhs, Expr rhs, Expr candidate, int[] lhsPackedAxes, int[] rhsPackedAxes, int[] lhsLanes, int[] rhsLanes)
    {
        var rets = new List<Expr>();
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var outShape = candidate.CheckedShape;
        if (lhsPackedAxes.Length == 0 && rhsPackedAxes.Length == 0)
        {
            return rets;
        }

        var alignedLhsPackedAxes = lhsPackedAxes.Select(a => a + outShape.Rank - lhsShape.Rank).ToArray();
        var alignedRhsPackedAxes = rhsPackedAxes.Select(a => a + outShape.Rank - rhsShape.Rank).ToArray();
        if (lhsPackedAxes.Any(a => lhsShape[a] is { IsFixed: true, FixedValue: var d } && d == 1)
        || rhsPackedAxes.Any(a => rhsShape[a] is { IsFixed: true, FixedValue: var d } && d == 1))
        {
            return rets;
        }

        var alignedLhsShape = Enumerable.Repeat(new DimConst(1), outShape.Rank - lhsShape.Rank).Concat(lhsShape).ToArray();
        var alignedRhsShape = Enumerable.Repeat(new DimConst(1), outShape.Rank - rhsShape.Rank).Concat(rhsShape).ToArray();
        if (alignedLhsPackedAxes.Any(a => alignedRhsShape[a] is { IsFixed: true, FixedValue: var d } && d != 1 && !alignedRhsPackedAxes.Contains(a))
        || alignedRhsPackedAxes.Any(a => alignedLhsShape[a] is { IsFixed: true, FixedValue: var d } && d != 1 && !alignedLhsPackedAxes.Contains(a)))
        {
            return rets;
        }

        var packedLhs = IR.F.Tensors.Pack(PackUtility.PadForPack(lhs, lhsShape, lhsPackedAxes, lhsLanes, 0f, out var lhsPadNums), lhsLanes, lhsPackedAxes);
        var packedRhs = IR.F.Tensors.Pack(PackUtility.PadForPack(rhs, rhsShape, rhsPackedAxes, rhsLanes, 0f, out var rhsPadNums), rhsLanes, rhsPackedAxes);

        var compare = IR.F.Math.Compare(op.CompareOp, packedLhs, packedRhs);
        var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(compare, lhsLanes.Length >= rhsLanes.Length ? lhsLanes : rhsLanes, lhsPackedAxes.Length >= rhsPackedAxes.Length ? alignedLhsPackedAxes : alignedRhsPackedAxes), candidate.CheckedShape, lhsPackedAxes.Length >= rhsPackedAxes.Length ? lhsPadNums! : rhsPadNums!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.Math.Compare)result["target"];
        var lhs = (Expr)result["lhs"];
        var rhs = (Expr)result["rhs"];
        var candidate = (Expr)result[Pattern];
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var lhsLaneSize = Lane / lhs.CheckedDataType.SizeInBytes;
        var rhsLaneSize = Lane / rhs.CheckedDataType.SizeInBytes;

        foreach (var arr in new[] { GeneratePackAxes(lhsShape), GeneratePackAxes(rhsShape) }.CartesianProduct())
        {
            var lhsPackedAxes = arr.First();
            var rhsPackedAxes = arr.Skip(1).First();
            if (lhsPackedAxes.Length <= Rank && rhsPackedAxes.Length <= Rank)
            {
                rets.AddRange(AddCandidate(op, lhs, rhs, candidate, lhsPackedAxes, rhsPackedAxes, Enumerable.Repeat(lhsLaneSize, lhsPackedAxes.Length).ToArray(), Enumerable.Repeat(rhsLaneSize, rhsPackedAxes.Length).ToArray()));
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
      "call",
      _ => true,
      IsWildcard("input", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = IsFloat() & !IsVector() });

    public static List<Expr> AddCandidate(Call call, Expr input, int[] packedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var op = (IR.Tensors.Cast)call.Target;
        var inShape = input.CheckedShape;
        if (packedAxes.Length == 0)
        {
            return rets;
        }

        var packedInput = IR.F.Tensors.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

        var scale = 1f * call.CheckedDataType.SizeInBytes / input.CheckedDataType.SizeInBytes;
        var outLanes = lanes.Select(l => (int)(l / scale)).ToArray();
        var newType = new VectorType(op.NewType, outLanes);

        var cast = IR.F.Tensors.Cast(packedInput, newType, op.CastMode, packedAxes);
        var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(cast, outLanes, packedAxes), inShape, padsInput!);
        if (cast.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var call = (Call)result["call"];
        var input = (Expr)result["input"];
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(call, input, [i], [laneSize]));
            for (int j = i + 1; j < input.CheckedShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    rets.AddRange(AddCandidate(call, input, [i, j], [laneSize, laneSize]));
                }
            }
        }

        return rets;
    }
}

public sealed class PackExpand : PackRule
{
    public PackExpand(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsExpand(
      "target",
      "call",
      _ => true,
      IsWildcard("input", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = !IsVector() },
      IsFixedShape("shape"));

    public static List<Expr> AddCandidate(Call call, Expr input, long[] shape, int[] packedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var op = (IR.Tensors.Expand)call.Target;
        var inShape = input.CheckedShape;
        if (packedAxes.Length == 0)
        {
            return rets;
        }

        if (packedAxes.Any(a => inShape[a] is { IsFixed: true, FixedValue: var d } && d == 1))
        {
            return rets;
        }

        var packedInput = IR.F.Tensors.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

        // only support shape >= input.shape
        var packedNewShape = shape.ToArray();
        foreach (var (lane, axis) in lanes.Zip(packedAxes))
        {
            packedNewShape[axis] = MathUtility.CeilDiv(packedNewShape[axis], lane);
        }

        var expand = IR.F.Tensors.Expand(packedInput, packedNewShape);
        var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(expand, lanes, packedAxes), call.CheckedShape, padsInput!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var call = (Call)result["call"];
        var input = (Expr)result["input"];
        var shape = ((RankedShape)result["shape"]).ToValueArray();
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(call, input, shape, [i], [laneSize]));
        }

        return rets;
    }
}

public sealed class PackWhere : PackRule
{
    public PackWhere(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsWhere(
        "target",
        "call",
        _ => true,
        IsWildcard("condition", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = !IsVector() },
        IsWildcard("lhs", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = !IsVector() },
        IsWildcard("rhs", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = !IsVector() });

    public static List<Expr> AddCandidate(Expr condition, Expr lhs, Expr rhs, Expr candidate, int[] conditionPackedAxes, int[] lhsPackedAxes, int[] rhsPackedAxes, int[] conditionLanes, int[] lhsLanes, int[] rhsLanes)
    {
        var rets = new List<Expr>();
        var conditionShape = condition.CheckedShape;
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var outShape = candidate.CheckedShape;
        if (conditionPackedAxes.Length == 0 && lhsPackedAxes.Length == 0 && rhsPackedAxes.Length == 0)
        {
            return rets;
        }

        var alignedConditionPackedAxes = conditionPackedAxes.Select(a => a + outShape.Rank - conditionShape.Rank).ToArray();
        var alignedLhsPackedAxes = lhsPackedAxes.Select(a => a + outShape.Rank - lhsShape.Rank).ToArray();
        var alignedRhsPackedAxes = rhsPackedAxes.Select(a => a + outShape.Rank - rhsShape.Rank).ToArray();
        var union = alignedConditionPackedAxes.Union(alignedLhsPackedAxes).Union(alignedRhsPackedAxes).ToArray();
        if (union.Length > conditionPackedAxes.Length && union.Length > lhsPackedAxes.Length && union.Length > rhsPackedAxes.Length)
        {
            return rets;
        }

        if (conditionPackedAxes.Any(a => conditionShape[a] is { IsFixed: true, FixedValue: var d } && d == 1)
            || lhsPackedAxes.Any(a => lhsShape[a] is { IsFixed: true, FixedValue: var d } && d == 1)
            || rhsPackedAxes.Any(a => rhsShape[a] is { IsFixed: true, FixedValue: var d } && d == 1))
        {
            return rets;
        }

        var alignedCondShape = Enumerable.Repeat(new DimConst(1), outShape.Rank - conditionShape.Rank).Concat(conditionShape).ToArray();
        var alignedLhsShape = Enumerable.Repeat(new DimConst(1), outShape.Rank - lhsShape.Rank).Concat(lhsShape).ToArray();
        var alignedRhsShape = Enumerable.Repeat(new DimConst(1), outShape.Rank - rhsShape.Rank).Concat(rhsShape).ToArray();
        if (alignedConditionPackedAxes.Any(a => alignedLhsShape[a] is { IsFixed: true, FixedValue: var d } && d != 1 && !alignedLhsPackedAxes.Contains(a))
        || alignedConditionPackedAxes.Any(a => alignedRhsShape[a] is { IsFixed: true, FixedValue: var d } && d != 1 && !alignedRhsPackedAxes.Contains(a))
        || alignedLhsPackedAxes.Any(a => alignedCondShape[a] is { IsFixed: true, FixedValue: var d } && d != 1 && !alignedConditionPackedAxes.Contains(a))
        || alignedLhsPackedAxes.Any(a => alignedRhsShape[a] is { IsFixed: true, FixedValue: var d } && d != 1 && !alignedRhsPackedAxes.Contains(a))
        || alignedRhsPackedAxes.Any(a => alignedCondShape[a] is { IsFixed: true, FixedValue: var d } && d != 1 && !alignedConditionPackedAxes.Contains(a))
        || alignedRhsPackedAxes.Any(a => alignedLhsShape[a] is { IsFixed: true, FixedValue: var d } && d != 1 && !alignedLhsPackedAxes.Contains(a)))
        {
            return rets;
        }

        var packedCondition = IR.F.Tensors.Pack(PackUtility.PadForPack(condition, conditionShape, conditionPackedAxes, conditionLanes, 0f, out var conditionPadNums), conditionLanes, conditionPackedAxes);
        var packedLhs = IR.F.Tensors.Pack(PackUtility.PadForPack(lhs, lhsShape, lhsPackedAxes, lhsLanes, 0f, out var lhsPadNums), lhsLanes, lhsPackedAxes);
        var packedRhs = IR.F.Tensors.Pack(PackUtility.PadForPack(rhs, rhsShape, rhsPackedAxes, rhsLanes, 0f, out var rhsPadNums), rhsLanes, rhsPackedAxes);

        var compare = IR.F.Tensors.Where(packedCondition, packedLhs, packedRhs);
        var allLanes = new[] { conditionLanes, lhsLanes, rhsLanes };
        var maxIndex = Enumerable.Range(0, allLanes.Length).OrderByDescending(i => allLanes[i].Length).First();
        var outLanes = allLanes[maxIndex];
        var outPackAxes = new[] { alignedConditionPackedAxes, alignedLhsPackedAxes, alignedRhsPackedAxes }.ElementAt(maxIndex);
        var outPadNums = new[] { conditionPadNums, lhsPadNums, rhsPadNums }.ElementAt(maxIndex);
        var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(compare, outLanes, outPackAxes), candidate.CheckedShape, outPadNums);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var condition = (Expr)result["condition"];
        var lhs = (Expr)result["lhs"];
        var rhs = (Expr)result["rhs"];
        var candidate = (Expr)result[Pattern];
        var conditionShape = condition.CheckedShape;
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var lhsLaneSize = Lane / lhs.CheckedDataType.SizeInBytes;
        var rhsLaneSize = Lane / rhs.CheckedDataType.SizeInBytes;

        // not supoort different lane size.
        var conditionLaneSize = lhsLaneSize;

        foreach (var arr in new[] { GeneratePackAxes(conditionShape), GeneratePackAxes(lhsShape), GeneratePackAxes(rhsShape) }.CartesianProduct())
        {
            var conditionPackedAxes = arr.First();
            var lhsPackedAxes = arr.Skip(1).First();
            var rhsPackedAxes = arr.Skip(2).First();
            if (conditionPackedAxes.Length <= Rank && lhsPackedAxes.Length <= Rank && rhsPackedAxes.Length <= Rank)
            {
                rets.AddRange(AddCandidate(condition, lhs, rhs, candidate, conditionPackedAxes, lhsPackedAxes, rhsPackedAxes, Enumerable.Repeat(conditionLaneSize, conditionPackedAxes.Length).ToArray(), Enumerable.Repeat(lhsLaneSize, lhsPackedAxes.Length).ToArray(), Enumerable.Repeat(rhsLaneSize, rhsPackedAxes.Length).ToArray()));
            }
        }

        return rets;
    }
}

public sealed class PackGather : PackRule
{
    public PackGather(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsGather(
        "target",
        "call",
        _ => true,
        IsWildcard("input", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = !IsVector() },
        IsWildcard("index"));

    public static List<Expr> AddCandidate(Call call, Expr input, Expr index, int[] packedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var op = (IR.Tensors.Gather)call.Target;
        var axis = op.Axis;
        var inShape = input.CheckedShape;
        if (packedAxes.Length == 0 || packedAxes.Contains(axis))
        {
            return rets;
        }

        var packedInput = IR.F.Tensors.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

        var gather = IR.F.Tensors.Gather(packedInput, axis, index);
        var unpackAxes = packedAxes.Select(a => a < axis ? a : a + index.CheckedShape.Rank - 1).ToArray();
        var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(gather, lanes, unpackAxes), call.CheckedShape, padsInput!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var call = (Call)result["call"];
        var input = (Expr)result["input"];
        var index = (Expr)result["index"];
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(call, input, index, [i], [laneSize]));
        }

        return rets;
    }
}

public sealed class PackScatterND : PackRule
{
    public PackScatterND(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsScatterND(
        "target",
        "call",
        _ => true,
        IsWildcard("input", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = !IsVector() },
        IsTensorConst("indices"),
        IsWildcard("updates", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = !IsVector() });

    public static List<Expr> AddCandidate(Call call, Expr input, TensorConst indices, Expr updates, int[] packedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var op = (IR.Tensors.ScatterND)call.Target;
        var inShape = input.CheckedShape;
        var indicesShape = indices.CheckedShape.ToValueArray();
        if (packedAxes.Length == 0 || Enumerable.Range(0, (int)indicesShape[^1]).Intersect(packedAxes).Any())
        {
            return rets;
        }

        var packedInput = IR.F.Tensors.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);
        var updatesPackedAxes = packedAxes.Select(a => a - (inShape.Rank - updates.CheckedShape.Rank)).ToArray();
        var packedUpdates = IR.F.Tensors.Pack(PackUtility.PadForPack(updates, updates.CheckedShape, updatesPackedAxes, lanes, 0f, out var padsUpdates), lanes, updatesPackedAxes);

        var scatter_nd = IR.F.Tensors.ScatterND(packedInput, indices, packedUpdates);
        var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(scatter_nd, lanes, packedAxes), inShape, padsInput!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var call = (Call)result["call"];
        var input = (Expr)result["input"];
        var indices = (TensorConst)result["indices"];
        var updates = (Expr)result["updates"];
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(call, input, indices, updates, [i], [laneSize]));
        }

        return rets;
    }
}

public class PackSoftmax : PackRule
{
    public PackSoftmax(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsSoftmax(
      "target",
      IsWildcard("input") with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("axis") with { TypePattern = IsIntegralScalar() });

    public static List<Expr> AddCandidate(Expr input, int axis, int[] packedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var inShape = input.CheckedShape;
        var packed = IR.F.Tensors.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, float.NegativeInfinity, out var pads), lanes, packedAxes);
        var softmax = IR.F.NTT.PackedSoftmax(packed, axis, packedAxes);
        if (softmax.CheckedType is not InvalidType)
        {
            var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(softmax, lanes, packedAxes), inShape, pads);
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var input = (Expr)result["input"];
        var axis = (int)((DimConst)result["axis"]).FixedValue;
        var inShape = input.CheckedShape;
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < inShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(input, axis, new[] { i }, new[] { laneSize }));
            for (int j = i + 1; j < inShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    rets.AddRange(AddCandidate(input, axis, new[] { i, j }, new[] { laneSize, laneSize }));
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
      IsWildcard("input") with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("scale") with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("bias") with { TypePattern = IsFloat() & !IsVector() });

    public static List<Expr> AddCandidate(LayerNorm op, Expr input, Expr scale, Expr bias, int[] packedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var inShape = input.CheckedShape;
        var pshape = scale.CheckedShape;
        var packedInput = IR.F.Tensors.Pack(PackUtility.PadForPack(input, inShape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

        var pAxes = packedAxes.Where(i => i >= op.Axis).Select(i => i - op.Axis).ToArray();
        var packedScale = PackUtility.PadForPack(scale, pshape, pAxes, lanes, 0f, out var padsScale);
        if (pAxes.Length > 0)
        {
            packedScale = IR.F.Tensors.Pack(packedScale, Enumerable.Repeat(lanes[0], pAxes.Length).ToArray(), pAxes);
        }

        var packedBias = PackUtility.PadForPack(bias, pshape, pAxes, lanes, 0f, out var padsBias);
        if (pAxes.Length > 0)
        {
            packedBias = IR.F.Tensors.Pack(packedBias, Enumerable.Repeat(lanes[0], pAxes.Length).ToArray(), pAxes);
        }

        var layernorm = IR.F.NTT.PackedLayerNorm(packedInput, packedScale, packedBias, op.Axis, op.Epsilon, op.UseMean, packedAxes, new RankedShape(padsInput));

        if (layernorm.CheckedType is not InvalidType)
        {
            var post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(layernorm, lanes, packedAxes), inShape, padsInput);
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.NN.LayerNorm)result["target"];
        var input = (Expr)result["input"];
        var scale = (Expr)result["scale"];
        var bias = (Expr)result["bias"];
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(op, input, scale, bias, new[] { i }, new[] { laneSize }));
            for (int j = i + 1; j < input.CheckedShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    rets.AddRange(AddCandidate(op, input, scale, bias, new[] { i, j }, new[] { laneSize, laneSize }));
                }
            }
        }

        return rets;
    }
}

[RuleGenerator]
public sealed partial class FoldPackUnpack : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = PatternMatch.F.Tensors.IsPack("pack", "caller", _ => true, PatternMatch.F.Tensors.IsUnpack("unpack", "callee", _ => true, IsWildcard("input")));

    private Expr? GetReplace(IR.Tensors.Pack pack, IR.Tensors.Unpack unpack, Expr input)
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
    public override Pattern Pattern { get; } = PatternMatch.F.Tensors.IsPack("pack", "caller", _ => true, PatternMatch.F.Tensors.IsConcat("concat", _ => true, IsTuple("tuple", IsVArgsRepeat("fileds", exprs =>
        {
            var patterns = new Pattern[exprs.Length];
            for (int i = 0; i < exprs.Length; i++)
            {
                patterns[i] = PatternMatch.F.Tensors.IsUnpack($"unpack_{i}", $"callee_{i}", _ => true, IsWildcard($"input_{i}"));
            }

            return patterns;
        }))));

    private Expr? GetReplace(IR.Tensors.Pack pack, IR.Tensors.Concat concat, IReadOnlyList<BaseExpr> fileds, IMatchResult result)
    {
        var inputs = new Expr[fileds.Count];
        for (int i = 0; i < fileds.Count; i++)
        {
            var unpack = (IR.Tensors.Unpack)result[$"unpack_{i}"];
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
    public override Pattern Pattern { get; } = PatternMatch.F.NTT.IsPackedMatMul("matmul", "caller", m => m.RhsPackedAxes.Count == 2 && m.TransposeB == false, IsWildcard("lhs"), PatternMatch.F.Tensors.IsPack("rhsPack", "callee", p => p.Axes.Count == 2 && p.Lanes.Count == 2, PatternMatch.F.Tensors.IsTranspose("trans", "rhs", IsWildcard("transInput"), IsFixedShape("perm") /* IsAlt(IsTensorConst("rhs"), PatternMatch.F.Tensors.IsTranspose("trans", "rhs", IsWildcard("transInput"), IsTensorConst("perm")) */)));

    private Expr? GetReplace(IR.NTT.PackedMatMul matmul, Expr lhs, IR.Tensors.Pack rhsPack, Expr transInput, int[] perm, IMatchResult result)
    {
        // note can't enable transpose const b, because const folding will be very solw.
        var inputsShape = transInput.CheckedShape.ToValueArray();
        var tperm = Enumerable.Range(0, inputsShape.Length).ToArray();
        (tperm[^2], tperm[^1]) = (tperm[^1], tperm[^2]);
        if (tperm.SequenceEqual(perm))
        {
            var npack = IR.F.Tensors.Pack(transInput, [rhsPack.Lanes[1], rhsPack.Lanes[0]], [rhsPack.Axes[1], rhsPack.Axes[0]]);
            var newMatmul = new IR.NTT.PackedMatMul(matmul.OutputDataType, matmul.LhsPackedAxes, new[] { matmul.RhsPackedAxes[1], matmul.RhsPackedAxes[0] }, false, true, matmul.FusedReduce);
            return new Call(newMatmul, lhs, npack);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class FoldNopPack : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = PatternMatch.F.Tensors.IsPack("pack", "call", _ => true, IsWildcard("input"));

    private Expr? GetReplace(IR.Tensors.Pack pack, Expr input)
    {
        if (pack.Axes.Count == 0)
        {
            return input;
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class FoldNopUnpack : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = PatternMatch.F.Tensors.IsUnpack("unpack", "call", _ => true, IsWildcard("input"));

    private Expr? GetReplace(IR.Tensors.Unpack unpack, Expr input)
    {
        if (unpack.Axes.Count == 0)
        {
            return input;
        }

        return null;
    }
}
