// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Reactive;
using DryIoc;
using NetFabric.Hyperlinq;
using Nncase.Evaluator.Tensors;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Passes.BufferSchedule;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Tile;

public enum ConditionKind
{
    Norm,
    Tail,
}

public record BucketCondition(ConditionKind Bid, ConditionKind Tid, ConditionKind BidTid)
{
}

public sealed class NodeInfo : IDisposable
{
    private readonly ExprPinner _pinner;
    private readonly TIR.Buffer? _buffer;

    public NodeInfo(TIR.Buffer? buffer, int[] tileShape, int[] outShape)
    {
        _buffer = buffer;
        TileShape = tileShape;
        OutShape = outShape;
        if (_buffer is not null)
        {
            _pinner = new ExprPinner(_buffer);
        }
        else
        {
            _pinner = new ExprPinner();
        }
    }

    public TIR.Buffer Buffer => _buffer!;

    public IReadOnlyList<int> OutShape { get; }

    public int[] TileShape { get; set; }

    public void Dispose() => _pinner.Dispose();
}

public sealed record TileFragment(BucketCondition Condition, IReadOnlyDictionary<Expr, NodeInfo> TileMap)
{
}

public sealed class FusionChecker
{
    private readonly List<KeyValuePair<Expr, int[]>> _initTileList;
    private IReadOnlyList<TileFragment>? _checkedResult;

    public FusionChecker(List<KeyValuePair<Expr, int[]>> initTileList)
    {
        _initTileList = initTileList;
    }

    public IReadOnlyList<TileFragment> CheckedResult => _checkedResult!;

    public IReadOnlyList<TileFragment> Check(Expr root)
    {
        if (_checkedResult is not null)
        {
            return _checkedResult;
        }

        var (buckets, conditions) = GetSplitBuckets();
        var tileMaps = new Dictionary<Expr, NodeInfo>[buckets.Count];

        for (var b = 0; b < buckets.Count; b++)
        {
            var bucket = buckets[b];
            Dictionary<Expr, NodeInfo> tileMap = new();

            var updatedTileShape = _initTileList.Last().Value.ToArray();
            if (_initTileList.Any(kv => kv.Key is Call { Target: MatMul }))
            {
                var candidateKs = GetCandidateKs(bucket);

                // search k first
                int finalK = 0;
                for (var k = 0; k < candidateKs.Count; k++)
                {
                    tileMap.Clear();
                    tileMap.Add(root, new(null!, updatedTileShape, bucket[root]));
                    Visit((Call)root, tileMap, bucket, candidateKs, k);
                    var ok = TryAllocate(tileMap, bucket);
                    if (ok)
                    {
                        tileMaps[b] = tileMap.ToDictionary(kv => kv.Key, kv => kv.Value);
                        finalK = k;
                    }
                    else
                    {
                        break;
                    }
                }

                for (var r = root.CheckedShape.Rank - 1; r >= 0; r--)
                {
                    if (_initTileList.Last().Value[r] == 32)
                    {
                        tileMap.Clear();
                        while (true)
                        {
                            tileMap!.Add(root, new NodeInfo(null!, updatedTileShape, bucket[root]));
                            Visit((Call)root, tileMap, bucket, candidateKs, finalK);
                            var ok = TryAllocate(tileMap, bucket);
                            if (ok)
                            {
                                tileMaps[b] = tileMap.ToDictionary(kv => kv.Key, kv => kv.Value);
                                if (updatedTileShape[r] + 32 > bucket[root][r])
                                {
                                    break;
                                }

                                updatedTileShape[r] += 32;
                            }
                            else
                            {
                                updatedTileShape[r] -= 32;
                                break;
                            }

                            tileMap.Clear();
                        }
                    }
                }
            }
            else
            {
                for (var r = root.CheckedShape.Rank - 1; r >= 0; r--)
                {
                    var incr = r == root.CheckedShape.Rank - 1 ? 32 : 1;
                    tileMap.Clear();
                    while (true)
                    {
                        tileMap.Add(root, new(null!, updatedTileShape, bucket[root]));
                        Visit((Call)root, tileMap, bucket, new());
                        var ok = TryAllocate(tileMap, bucket);
                        if (ok)
                        {
                            tileMaps[b] = tileMap.ToDictionary(kv => kv.Key, kv => kv.Value);
                            if (updatedTileShape[r] + incr > bucket[root][r])
                            {
                                break;
                            }

                            updatedTileShape[r] += incr;
                        }
                        else
                        {
                            updatedTileShape[r] -= incr;
                            break;
                        }

                        tileMap.Clear();
                    }
                }
            }
        }

        for (int b = 0; b < buckets.Count; b++)
        {
            TryAllocate(tileMaps[b], buckets[b], true);
        }

        return _checkedResult = conditions.Zip(tileMaps).Select(p => new TileFragment(p.First, p.Second)).ToList();
    }

    private static List<Dictionary<Expr, int>> GetCandidateKs(Dictionary<Expr, int[]> bucket)
    {
        var allKs = new Dictionary<Expr, List<int>>();
        foreach (var kv in bucket)
        {
            if (kv.Key is Call { Target: MatMul op } call)
            {
                var k = bucket[call[op.Parameters.First()]].Last();
                var ks = new List<int>();
                for (int i = 32; i < k; i += 32)
                {
                    ks.Add(i);
                }

                ks.Add(k);
                allKs.Add(kv.Key, ks);
            }
        }

        IEnumerable<IEnumerable<KeyValuePair<Expr, int>>> ret = new[] { Enumerable.Empty<KeyValuePair<Expr, int>>() };
        foreach (var kvp in allKs)
        {
            ret = from seq in ret
                  from item in kvp.Value
                  select seq.Concat(new[] { new KeyValuePair<Expr, int>(kvp.Key, item) });
        }

        return ret.Select(seq => seq.ToDictionary(kv => kv.Key, kv => kv.Value)).ToList();
    }

    private (List<Dictionary<Expr, int[]>> Buckets, List<BucketCondition> Conditions) GetSplitBuckets()
    {
        var buckets = new Dictionary<BucketCondition, Dictionary<Expr, int[]>>();
        foreach (var s in GetCandidateBuckets())
        {
            buckets.Add(s, new());
        }

        foreach (var kv in _initTileList)
        {
            var ndSbp = ((DistributedType)kv.Key.CheckedType).NdSBP;
            var hierarchy = ((DistributedType)kv.Key.CheckedType).Placement.Hierarchy;
            var divided = Enumerable.Range(0, ndSbp.Count).Where(i => ndSbp[i] is SBPSplit).Select(i => (((SBPSplit)ndSbp[i]).Axis, hierarchy[i])).ToArray();
            var dividedSlice = DistributedUtility.TryGetNonUniformDividedSlice((DistributedType)kv.Key.CheckedType);
            if (dividedSlice.Count == 1)
            {
                foreach (BucketCondition s in GetCandidateBuckets())
                {
                    buckets[s].Add(kv.Key, dividedSlice[0]);
                }
            }
            else
            {
                switch (divided.Length)
                {
                    case 1 when hierarchy[0] == divided[0].Item2:
                        foreach (BucketCondition s in Enum.GetValues(typeof(BucketCondition)))
                        {
                            if (s is BucketCondition { Bid: ConditionKind.Norm })
                            {
                                buckets[s].Add(kv.Key, dividedSlice[0]);
                            }
                            else
                            {
                                buckets[s].Add(kv.Key, dividedSlice[1]);
                            }
                        }

                        break;
                    case 1 when hierarchy[1] == divided[0].Item2:
                        foreach (BucketCondition s in GetCandidateBuckets())
                        {
                            if (s is BucketCondition { Tid: ConditionKind.Norm })
                            {
                                buckets[s].Add(kv.Key, dividedSlice[0]);
                            }
                            else
                            {
                                buckets[s].Add(kv.Key, dividedSlice[1]);
                            }
                        }

                        break;
                    case 2 when divided[0].Axis == divided[1].Axis:
                        foreach (BucketCondition s in GetCandidateBuckets())
                        {
                            if (s is BucketCondition { BidTid: ConditionKind.Norm })
                            {
                                buckets[s].Add(kv.Key, dividedSlice[0]);
                            }
                            else
                            {
                                buckets[s].Add(kv.Key, dividedSlice[1]);
                            }
                        }

                        break;
                    case 2 when divided[0].Axis != divided[1].Axis:
                        if (dividedSlice.Count == 2)
                        {
                            if (kv.Key.CheckedShape[divided[0].Axis].FixedValue % hierarchy[0] == 0)
                            {
                                foreach (BucketCondition s in GetCandidateBuckets())
                                {
                                    if (s is BucketCondition { Tid: ConditionKind.Norm })
                                    {
                                        buckets[s].Add(kv.Key, dividedSlice[0]);
                                    }
                                    else
                                    {
                                        buckets[s].Add(kv.Key, dividedSlice[1]);
                                    }
                                }
                            }
                            else
                            {
                                foreach (BucketCondition s in GetCandidateBuckets())
                                {
                                    if (s is BucketCondition { BidTid: ConditionKind.Norm })
                                    {
                                        buckets[s].Add(kv.Key, dividedSlice[0]);
                                    }
                                    else
                                    {
                                        buckets[s].Add(kv.Key, dividedSlice[1]);
                                    }
                                }
                            }
                        }

                        if (dividedSlice.Count == 4)
                        {
                            foreach (BucketCondition s in GetCandidateBuckets())
                            {
                                if (s is BucketCondition { Bid: ConditionKind.Norm, Tid: ConditionKind.Norm })
                                {
                                    buckets[s].Add(kv.Key, dividedSlice[0]);
                                }
                                else if (s is BucketCondition { Bid: ConditionKind.Norm, Tid: ConditionKind.Tail })
                                {
                                    buckets[s].Add(kv.Key, dividedSlice[1]);
                                }
                                else if (s is BucketCondition { Bid: ConditionKind.Tail, Tid: ConditionKind.Norm })
                                {
                                    buckets[s].Add(kv.Key, dividedSlice[2]);
                                }
                                else
                                {
                                    buckets[s].Add(kv.Key, dividedSlice[3]);
                                }
                            }
                        }

                        break;
                    default:
                        throw new NotImplementedException("Not support split");
                }
            }
        }

        List<Dictionary<Expr, int[]>> ret = new();
        List<BucketCondition> conditions = new();
        foreach (BucketCondition s in GetCandidateBuckets())
        {
            var bucket = buckets[s];
            bool redundant = false;
            foreach (var b in ret)
            {
                if (bucket.All(kv => kv.Value.SequenceEqual(b[kv.Key])))
                {
                    redundant = true;
                }

                if (redundant)
                {
                    break;
                }
            }

            if (!redundant)
            {
                conditions.Add(s);
                ret.Add(bucket);
            }
        }

        return (ret, conditions);
    }

    private IEnumerable<BucketCondition> GetCandidateBuckets() =>
        new[] {
            new[] { ConditionKind.Norm, ConditionKind.Tail },
            new[] { ConditionKind.Norm, ConditionKind.Tail },
            new[] { ConditionKind.Norm, ConditionKind.Tail },
        }.CartesianProduct().
        Select(p => p.ToArray()).
        Select(a => new BucketCondition(a[0], a[1], a[2]));

    private bool TryAllocate(Dictionary<Expr, NodeInfo> tileMap, Dictionary<Expr, int[]> bucket, bool finalAllocate = false)
    {
        var tileList = new List<KeyValuePair<Expr, NodeInfo>>();
        var exprs = ExprCollector.Collect(_initTileList.Last().Key).Where(e => e is not Op);
        foreach (var expr in exprs)
        {
            tileList.Add(new(expr, tileMap[expr]));
        }

        var tileBuffer = TryAllocate(tileList, bucket, finalAllocate);
        if (tileBuffer.Count > 0)
        {
            foreach (var kv in tileBuffer)
            {
                tileMap[kv.Key] = new NodeInfo(kv.Value, tileMap[kv.Key].TileShape, tileMap[kv.Key].OutShape.ToArray());
            }

            return true;
        }

        return false;
    }

    private Dictionary<Expr, TIR.Buffer> TryAllocate(List<KeyValuePair<Expr, NodeInfo>> tileList, Dictionary<Expr, int[]> bucket, bool finalAllocate = false)
    {
        // TODO:
        // 1. 支持不同数据类型的检查
        // 2. 支持weights和数据采用不一样的buffer，可以考虑按pass load weights
        // 3. 支持不同层的weights复用或者不复用等
        // 4. 支持线程数可配
        // 5. 如果切K，partial sum 要考虑扩大尺寸
        // 6. cache search的结果，返回时直接输出最终的buffer
        Dictionary<Expr, ScheduledBuffer> lifenessMap = new();

        void UpdateLifeness(int start, Expr expr, TIR.Buffer buffer, bool updateEnd)
        {
            lifenessMap.Add(expr, new ScheduledBuffer(new Lifeness(start, int.MaxValue), buffer));
            if (updateEnd)
            {
                foreach (var operand in expr.Operands.ToArray().Where(e => e is not Op))
                {
                    var userList = operand.Users.Where(u => u is Call).ToList();
                    if (userList.All(u => lifenessMap.ContainsKey(u)))
                    {
                        lifenessMap[operand].Lifeness.End = start + 1;
                    }
                }
            }
        }

        foreach (var (kv, i) in tileList.Select((kv, i) => (kv, i)))
        {
            var shape = kv.Value.TileShape;
            var strides = TensorUtilities.GetStrides(shape);
            var dtype = kv.Key.CheckedType switch
            {
                DistributedType d => d.TensorType.DType,
                TensorType te => te.DType,
                _ => throw new NotSupportedException("Not support type"),
            };

            var location = kv.Key switch
            {
                TensorConst { ValueType: DistributedType } => MemoryLocation.Rdata,
                Var => MemoryLocation.Input,
                Call { Target: IR.CPU.Store } => MemoryLocation.Output,
                _ => MemoryLocation.L2Data,
            };

            var bfname = kv.Key switch
            {
                Call c => c.Target.GetType().ToString().Split(".")[^1],
                Var v => v.Name,
                Const c => "cons",
                _ => throw new NotSupportedException(),
            }

            + i.ToString();
            Expr start = location switch
            {
                MemoryLocation.L2Data => IR.None.Default,
                MemoryLocation.Rdata => IR.F.Buffer.DDrOf(kv.Key),
                _ => TIR.F.CPU.PtrOf(bfname, kv.Key.CheckedDataType),
            };

            if (location is MemoryLocation.Input or MemoryLocation.Output)
            {
                shape = bucket[kv.Key];
                strides = TensorUtilities.GetStrides(shape);
            }

            Expr size;
            if (shape.Length == 0)
            {
                size = dtype.SizeInBytes;
            }
            else
            {
                size = shape[0] * strides[0] * dtype.SizeInBytes;
            }

            var memSpan = new MemSpan(start, size, location);
            var buffer = new TIR.Buffer(bfname, dtype, memSpan, shape.Select(s => (Expr)s).ToArray(), strides.Select(s => (Expr)s).ToArray());
            UpdateLifeness(i, kv.Key, buffer, location == MemoryLocation.L2Data);
        }

        foreach (var kv in lifenessMap)
        {
            if (kv.Value.Lifeness.End == int.MaxValue)
            {
                kv.Value.Lifeness.End = kv.Value.Lifeness.Start + 2;
            }
        }

        bool ok = SchedulerSolver.ScheduleByCpModel(lifenessMap, true, 1f, out var scheduledBufferMap);
        var ret = new Dictionary<Expr, TIR.Buffer>();
        if (ok)
        {
            foreach (var (key, candidateSched) in lifenessMap)
            {
                if (scheduledBufferMap.TryGetValue(key, out var schedBuffer))
                {
                    ret.Add(key, schedBuffer.Buffer);
                }
                else
                {
                    ret.Add(key, candidateSched.Buffer);
                }
            }

            if (finalAllocate && Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Rewrite))
            {
                var scheduleResponse = new ScheduledResponse(scheduledBufferMap, ok);
                scheduleResponse.Dump("buffers", "auto");
            }
        }

        return ret;
    }

    private void Visit(Call expr, Dictionary<Expr, NodeInfo> tileMap, Dictionary<Expr, int[]> bucketMap, List<Dictionary<Expr, int>> candidateKs, int k = -1)
    {
        switch (expr.Target)
        {
            case IR.Math.MatMul op:
                VisitMatmul(op, expr, tileMap, bucketMap, candidateKs, k);
                break;
            case IR.Math.Unary or IR.CPU.Load or IR.CPU.Store:
                VisitIdenity(expr, tileMap, bucketMap, candidateKs, k);
                break;
            case IR.Math.Binary op:
                VisitBinary(op, expr, tileMap, bucketMap, candidateKs, k);
                break;
            default:
                throw new NotImplementedException("Not Implemented Op: " + expr.Target);
        }
    }

    private void VisitIdenity(Call call, Dictionary<Expr, NodeInfo> tileMap, Dictionary<Expr, int[]> bucketMap, List<Dictionary<Expr, int>> candidateKs, int k = -1)
    {
        var inTileShape = tileMap[call].TileShape;
        var input = call.Arguments[0];
        if (input is Var or TensorConst)
        {
            tileMap.Add(input, new(null!, inTileShape, bucketMap[input]));
        }
        else
        {
            if (tileMap.ContainsKey(input))
            {
                tileMap[input].TileShape = inTileShape.Select((s, i) => Math.Max(s, tileMap[input].TileShape[i])).ToArray();
            }
            else
            {
                tileMap.Add(input, new(null!, inTileShape, bucketMap[input]));
            }

            Visit((Call)input, tileMap, bucketMap, candidateKs, k);
        }
    }

    private void VisitMatmul(IR.Math.MatMul op, Call call, Dictionary<Expr, NodeInfo> tileMap, Dictionary<Expr, int[]> bucketMap, List<Dictionary<Expr, int>> candidateKs, int k)
    {
        var lhs = call.Arguments[0];
        var rhs = call.Arguments[1];

        var outTileShape = tileMap[call].TileShape;
        var inTileShapeA = Enumerable.Repeat(1, lhs.CheckedShape.Rank).ToArray();
        inTileShapeA[^2] = outTileShape[^2];
        inTileShapeA[^1] = candidateKs[k][call];
        var inTileShapeB = Enumerable.Repeat(1, rhs.CheckedShape.Rank).ToArray();
        inTileShapeB[^2] = candidateKs[k][call];
        inTileShapeB[^1] = outTileShape[^1];

        if (!(lhs is Var or TensorConst))
        {
            if (tileMap.ContainsKey(lhs))
            {
                tileMap[lhs].TileShape = inTileShapeA.Select((s, i) => Math.Max(s, tileMap[lhs].TileShape[i])).ToArray();
            }
            else
            {
                tileMap.Add(lhs, new(null!, inTileShapeA, bucketMap[lhs]));
            }

            Visit((Call)lhs, tileMap, bucketMap, candidateKs, k);
        }
        else
        {
            tileMap.Add(lhs, new(null!, inTileShapeA, bucketMap[lhs]));
        }

        if (!(rhs is Var or TensorConst))
        {
            if (tileMap.ContainsKey(rhs))
            {
                tileMap[rhs].TileShape = inTileShapeB.Select((s, i) => Math.Max(s, tileMap[rhs].TileShape[i])).ToArray();
            }
            else
            {
                tileMap.Add(rhs, new(null!, inTileShapeB, bucketMap[rhs]));
            }

            Visit((Call)rhs, tileMap, bucketMap, candidateKs, k);
        }
        else
        {
            tileMap.Add(rhs, new(null!, inTileShapeB, bucketMap[rhs]));
        }
    }

    private void VisitBinary(IR.Math.Binary op, Call call, Dictionary<Expr, NodeInfo> tileMap, Dictionary<Expr, int[]> bucketMap, List<Dictionary<Expr, int>> candidateKs, int k)
    {
        var lhs = call.Arguments[0];
        var rhs = call.Arguments[1];

        var outTileShape = tileMap[call].TileShape;
        var padLhs = outTileShape.Length - lhs.CheckedShape.Rank;
        var inTileShapeA = Enumerable.Range(0, lhs.CheckedShape.Rank).Select(i => lhs.CheckedShape[i].FixedValue == 1 ? 1 : outTileShape[i + padLhs]).ToArray();
        var padRhs = outTileShape.Length - rhs.CheckedShape.Rank;
        var inTileShapeB = Enumerable.Range(0, rhs.CheckedShape.Rank).Select(i => rhs.CheckedShape[i].FixedValue == 1 ? 1 : outTileShape[i + padRhs]).ToArray();

        if (!(lhs is Var or TensorConst))
        {
            if (tileMap.ContainsKey(lhs))
            {
                tileMap[lhs].TileShape = inTileShapeA.Select((s, i) => Math.Max(s, tileMap[lhs].TileShape[i])).ToArray();
            }
            else
            {
                tileMap.Add(lhs, new(null!, inTileShapeA, bucketMap[lhs]));
            }

            Visit((Call)lhs, tileMap, bucketMap, candidateKs, k);
        }
        else
        {
            tileMap.Add(lhs, new(null!, inTileShapeA, bucketMap[lhs]));
        }

        if (!(rhs is Var or TensorConst))
        {
            if (tileMap.ContainsKey(rhs))
            {
                tileMap[rhs].TileShape = inTileShapeB.Select((s, i) => Math.Max(s, tileMap[rhs].TileShape[i])).ToArray();
            }
            else
            {
                tileMap.Add(rhs, new(null!, inTileShapeB, bucketMap[rhs]));
            }

            Visit((Call)rhs, tileMap, bucketMap, candidateKs, k);
        }
        else
        {
            tileMap.Add(rhs, new(null!, inTileShapeB, bucketMap[rhs]));
        }
    }
}
