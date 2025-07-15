// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

public sealed class PagedAttentionEvaluator : ITypeInferencer<PagedAttention>, ICostEvaluator<PagedAttention>, IEvaluator<PagedAttention>
{
    public IRType Visit(ITypeInferenceContext context, PagedAttention target)
    {
        var q = context.CheckArgumentType<IRType>(target, PagedAttention.Q);
        var extra = context.CheckArgumentType<IRType>(target, PagedAttention.Extra);
        var scale = context.CheckArgumentType<TensorType>(target, PagedAttention.Scale);
        var kvcaches = context.CheckArgumentType<TensorType>(target, PagedAttention.KVCaches);

        return (q, extra) switch
        {
            (DistributedType dq, DistributedType dextra) => Visit(context, target, dq, dextra, scale, kvcaches),
            (TensorType tq, TensorType textra) => Visit(context, target, tq, textra, scale, kvcaches, out _),
            _ => new InvalidType("not support type"),
        };
    }

    public Cost Visit(ICostEvaluateContext context, PagedAttention target)
    {
        var qType = context.GetArgumentType<IRType>(target, PagedAttention.Q);
        var returnType = context.GetReturnType<IRType>();

        // cycles = softmax((q @ k^t) + mask) @ v.
        return new()
        {
            // todo kv cache
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(qType),

            // todo [CostFactorNames.CPUCycles].
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public IValue Visit(IEvaluateContext context, PagedAttention target)
    {
        var q = context.GetOrtArgumentValue(target, PagedAttention.Q);
        var kvCaches = context.GetArgumentValueAsTensor<Reference<IPagedAttentionKVCache>>(target, PagedAttention.KVCaches);
        var scale = context.GetOrtArgumentValue(target, PagedAttention.Scale); // must match to prim kv type.
        return RefPagedAttn(q, kvCaches, scale, target.LayerId, target.Layout).ToValue();
    }

    private static OrtKISharp.Tensor RefPagedAttn(OrtKISharp.Tensor query, Tensor<Reference<IPagedAttentionKVCache>> kvCaches, OrtKISharp.Tensor scale, int layerId, IRArray<AttentionDimKind> qlayout)
    {
        // TODO: Support DP
        if (kvCaches.Length != 1)
        {
            throw new ArgumentException($"kvCaches length {kvCaches.Length} != 1");
        }

        var cache = kvCaches.Single().Value;

        // unpack for q
        if (cache.Config.PackedAxes.Contains(PagedKVCacheDimKind.HeadDim))
        {
            query = query.Unpack(qlayout.IndexOf(AttentionDimKind.Dim));
        }

        if (cache.Config.PackedAxes.Contains(PagedKVCacheDimKind.NumKVHeads))
        {
            query = query.Unpack(qlayout.IndexOf(AttentionDimKind.Head));
        }

        // revert transpose
        if (!qlayout.SequenceEqual([AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim]))
        {
            var invPerm = qlayout.Zip(Enumerable.Range(0, qlayout.Count)).OrderBy(p => p.First).Select(p => (long)p.Second).ToArray();
            query = OrtKI.Transpose(query, invPerm);
        }

        var outputs = new List<OrtKISharp.Tensor>();
        long queryStart = 0;
        for (int seqId = 0; seqId < cache.NumSeqs; seqId++)
        {
            var seqLen = cache.SeqLen(seqId);
            var queryLen = seqLen - cache.ContextLen(seqId);

            // [query_len, num_heads, head_dim] [L,Hq,E]
            var q = OrtKI.Slice(query, new long[] { queryStart }, new long[] { queryStart + queryLen }, new long[] { 0L }, new long[] { 1L });

            var k = GatherKV(AttentionCacheKind.Key, cache, seqId, layerId, queryLen, seqLen, queryStart); // [num_heads, seq_len, head_dim] [H,S,E]
            if (k.Shape[0] != q.Shape[1])
            {
                var repeatCount = (int)(q.Shape[1] / k.Shape[0]);
                var indices = Enumerable.Range(0, (int)k.Shape[0]).
                    Select(i => Enumerable.Repeat(i, repeatCount)).
                    SelectMany(i => i).
                    Select(i => (long)i).
                    ToArray();
                k = OrtKI.Gather(k, indices, 0);
            }

            var attn = OrtKI.Einsum([q, k], "LHE,HSE->HLS") * scale; // [num_heads, query_len, seq_len] [H,L,S]

            // compute causal mask
            var attnBias = OrtKI.Expand(OrtKISharp.Tensor.FromScalar(0f), OrtKISharp.Tensor.MakeTensor([queryLen, seqLen]));
            var tempMask = OrtKISharp.Tensor.MakeTensor(Enumerable.Repeat(1.0f, (int)(queryLen * seqLen)).ToArray(), [queryLen, seqLen]);

            tempMask = OrtKI.Trilu(tempMask, seqLen - queryLen, 0);
            attnBias = OrtKI.Where(OrtKI.Equal(tempMask, OrtKISharp.Tensor.FromScalar(1.0f)), attnBias, OrtKI.Expand(OrtKISharp.Tensor.FromScalar(float.NegativeInfinity), OrtKISharp.Tensor.MakeTensor([queryLen, seqLen])));
            attn = attn + OrtKI.Cast(attnBias, (long)attn.DataType);

            attn = OrtKI.Softmax(attn, -1);

            var v = GatherKV(AttentionCacheKind.Value, cache, seqId, layerId, queryLen, seqLen, queryStart); // [num_heads, seq_len, head_dim] [H,S,E]
            if (v.Shape[0] != q.Shape[1])
            {
                var repeatCount = (int)(q.Shape[1] / v.Shape[0]);
                var indices = Enumerable.Range(0, (int)v.Shape[0]).
                    Select(i => Enumerable.Repeat(i, repeatCount)).
                    SelectMany(i => i).
                    Select(i => (long)i).
                    ToArray();
                v = OrtKI.Gather(v, indices, 0);
            }

            var output = OrtKI.Einsum([attn, v], "HLS,HSE->LHE"); // [query_len, num_heads, head_dim] [L,H,E]

            outputs.Add(output);
            queryStart += queryLen;
        }

        var concat_output = OrtKI.Concat(outputs.ToArray(), 0L); // concat at seqs

        // retranspose output
        if (!qlayout.SequenceEqual([AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim]))
        {
            concat_output = OrtKI.Transpose(concat_output, qlayout.Select(i => (long)i).ToArray());
        }

        // repack for output
        if (cache.Config.PackedAxes.Contains(PagedKVCacheDimKind.NumKVHeads))
        {
            concat_output = concat_output.Pack(cache.Config.Lanes[cache.Config.PackedAxes.IndexOf(PagedKVCacheDimKind.NumKVHeads)], qlayout.IndexOf(AttentionDimKind.Head));
        }

        if (cache.Config.PackedAxes.Contains(PagedKVCacheDimKind.HeadDim))
        {
            concat_output = concat_output.Pack(cache.Config.Lanes[cache.Config.PackedAxes.IndexOf(PagedKVCacheDimKind.HeadDim)], qlayout.IndexOf(AttentionDimKind.Dim));
        }

        return concat_output;
    }

    private static void GatherKVCore(IPagedAttentionKVCache cache, int numBlocksForSeq, int seqId, AttentionCacheKind cacheKind, int layerId, int seqLen, int blockSizeAxis, List<OrtKISharp.Tensor> caches)
    {
        for (int headId = 0; headId < cache.Config.NumKVHeads; headId++)
        {
            for (int i = 0; i < numBlocksForSeq; i++)
            {
                var blockId = cache.GetBlockId(seqId, i);

                var headIdCopy = headId;
                var blockIdCopy = blockId.AsContiguous(true);

                // process sharding axes.
                for (int shardId = 0; shardId < cache.Config.ShardingAxes.Count; shardId++)
                {
                    switch (cache.Config.ShardingAxes[shardId])
                    {
                        case PagedKVCacheDimKind.NumKVHeads when blockIdCopy[shardId] is -1L:
                            var headTile = cache.Config.NumKVHeads / (int)cache.LogicalCacheDimensions()[shardId];
                            blockIdCopy[shardId] = System.Math.DivRem(headId, headTile, out headIdCopy);
                            break;
                        case PagedKVCacheDimKind.NumBlocks when blockIdCopy[shardId] is not -1L:
                            break;
                        default:
                            throw new ArgumentOutOfRangeException(nameof(cache));
                    }
                }

                var block = cache.GetBlock(cacheKind, layerId, headIdCopy, blockIdCopy);
                var blockOrt = block.ToOrtTensor();

                // slice
                var validSlotCount = (int)System.Math.Min(seqLen - (i * cache.Config.BlockSize), cache.Config.BlockSize);
                if (validSlotCount < cache.Config.BlockSize)
                {
                    blockOrt = OrtKI.Slice(blockOrt, new long[] { 0L }, new long[] { validSlotCount }, new long[] { blockSizeAxis }, new long[] { 1 });
                }

                caches.Add(blockOrt);
            }
        }
    }

    private static OrtKISharp.Tensor GatherKV(AttentionCacheKind cacheKind, IPagedAttentionKVCache cache, int seqId, int layerId, long queryLen, long seqLen, long queryStart)
    {
        var caches = new List<OrtKISharp.Tensor>();

        // var blockIds = cache.GetBlockId(seqId);
        var numBlocksForSeq = MathUtility.CeilDiv(cache.SeqLen(seqId), cache.Config.BlockSize);

        // block layout is construct from `head dim, block_size`. but we don't know the concrete shape.
        var blockLayout = cache.Config.BlockLayout;
        var blockSizeAxis = blockLayout.IndexOf(PagedKVCacheDimKind.BlockSize);
        if (blockSizeAxis < 0)
        {
            throw new InvalidOperationException("block layout not contain block size");
        }

        var headDimAxis = blockLayout.IndexOf(PagedKVCacheDimKind.HeadDim);
        if (headDimAxis < 0)
        {
            throw new InvalidOperationException("block layout not contain head dim");
        }

        int totalKVHeads = cache.Config.NumKVHeads;

        GatherKVCore(cache, (int)numBlocksForSeq, seqId, cacheKind, layerId, (int)seqLen, blockSizeAxis, caches);

        // caches is (head * blocks) * [BlockLayout + lanes], concat at block size axis.
        var concatCache = OrtKI.Concat(caches.ToArray(), blockSizeAxis);

        // unpack head dim
        if (cache.Config.PackedAxes.Contains(PagedKVCacheDimKind.HeadDim))
        {
            concatCache = concatCache.Unpack(headDimAxis);
        }

        // transpose to [num_head * seq_len, head_dim]
        if (blockLayout is [PagedKVCacheDimKind.HeadDim, PagedKVCacheDimKind.BlockSize])
        {
            concatCache = OrtKI.Transpose(concatCache, [1, 0]);
        }

        var newShape = concatCache.Shape.ToList(); // split head and seq len dim.
        newShape.Insert(0, totalKVHeads);
        newShape[1] /= totalKVHeads;
        concatCache.Reshape(newShape.ToArray());
        return concatCache; // [num_heads, seq_len, head_dim]
    }

    private IRType Visit(ITypeInferenceContext context, PagedAttention target, TensorType q, IRType extra, TensorType scale, TensorType kvCaches, out PagedAttentionKVCacheType pagedAttentionKVCacheType)
    {
        pagedAttentionKVCacheType = null!;
        if (kvCaches.DType is not ReferenceType { ElemType: PagedAttentionKVCacheType kVCacheType })
        {
            return new InvalidType("kv cache type not support!");
        }

        pagedAttentionKVCacheType = kVCacheType;

        if (!scale.IsScalar || scale.DType is not PrimType primType)
        {
            return new InvalidType($"scale {scale} should be scalar");
        }

        if ((q.DType is PrimType qPrimType && qPrimType != primType) ||
            (q.DType is VectorType v && v.ElemType is PrimType vPrimType && vPrimType != primType))
        {
            return new InvalidType($"q {q.DType} and scale {scale.DType} should have same dtype");
        }

        return q;
    }

    private IRType Visit(ITypeInferenceContext context, PagedAttention target, DistributedType q, DistributedType extra, TensorType scale, TensorType kvCaches)
    {
        if (Visit(context, target, q.TensorType, extra, scale, kvCaches, out var kVCacheType) is InvalidType invalidType)
        {
            return invalidType;
        }

        // for xpu.
        if (q.Placement.Name == "cdyxt")
        {
            if (!extra.AxisPolicies.All(p => p is SBPBroadCast))
            {
                return new InvalidType("extra should be broadcast!");
            }

            if (!target.Layout.SequenceEqual([AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq]))
            {
                return new InvalidType("layout should be [head, dim, seq]");
            }

            if (kVCacheType.Config is not IPagedAttentionConfig config)
            {
                return new InvalidType("kv cache has not config!");
            }

            if (!config.CacheLayout.SequenceEqual([PagedKVCacheDimKind.NumBlocks, PagedKVCacheDimKind.NumLayers, PagedKVCacheDimKind.NumKVHeads, PagedKVCacheDimKind.KV, PagedKVCacheDimKind.HeadDim, PagedKVCacheDimKind.BlockSize]))
            {
                return new InvalidType("kv cache block layout not support!");
            }

            if (!config.PackedAxes.SequenceEqual([PagedKVCacheDimKind.HeadDim]))
            {
                return new InvalidType("kv cache pack axes not support!");
            }

            if ((config.Lanes[0] * config.KVPrimType.SizeInBytes) != 128)
            {
                return new InvalidType("kv cache packed lanes not support!");
            }

            if (!config.ShardingAxes.SequenceEqual([PagedKVCacheDimKind.NumKVHeads, PagedKVCacheDimKind.NumBlocks]))
            {
                return new InvalidType("kv cache sharding axes not support!");
            }

            if (!config.AxisPolicies[0].Axes.SequenceEqual([1]) || !config.AxisPolicies[1].Axes.SequenceEqual([2, 3]))
            {
                return new InvalidType("kv cache axis policies not support!");
            }

            // seq split at x, head split at die and y, please check head size > 2*4.
            var seqAxis = target.Layout.IndexOf(AttentionDimKind.Seq);
            var headAxis = target.Layout.IndexOf(AttentionDimKind.Head);
            var dimAxis = target.Layout.IndexOf(AttentionDimKind.Dim);
            if (q.AxisPolicies[seqAxis] is SBPSplit { Axes: [2] } &&
                q.AxisPolicies[headAxis] is SBPSplit { Axes: [1, 3] } &&
                q.AxisPolicies[dimAxis] is SBPBroadCast)
            {
                return q;
            }
        }
        else
        {
            // seq broadcast, head split.
            var seqAxis = target.Layout.IndexOf(AttentionDimKind.Seq);
            var headAxis = target.Layout.IndexOf(AttentionDimKind.Head);
            var dimAxis = target.Layout.IndexOf(AttentionDimKind.Dim);
            if (q.AxisPolicies[seqAxis] is SBPBroadCast &&
                q.AxisPolicies[headAxis] is SBPSplit &&
                q.AxisPolicies[dimAxis] is SBPBroadCast)
            {
                return q;
            }
        }

        return new InvalidType("not support distributed type");
    }
}
