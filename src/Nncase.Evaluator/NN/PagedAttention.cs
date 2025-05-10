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
        return (q, extra) switch
        {
            (DistributedType dq, DistributedType dextra) => Visit(context, target, dq, dextra),
            (TensorType tq, TensorType textra) => Visit(context, target, tq, textra),
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
        return RefPagedAttn(q, kvCaches, 1.0f, target.LayerId, target.QLayout).ToValue();
    }

    private static OrtKISharp.Tensor RefPagedAttn(OrtKISharp.Tensor query, Tensor<Reference<IPagedAttentionKVCache>> kvCaches, float scale, int layerId, IRArray<AttentionDimKind> qlayout)
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
            var q = OrtKI.Slice(query, new long[] { queryStart }, new long[] { queryStart + queryLen }, new long[] { 0L }, new long[] { 1L });

            q = q * OrtKI.Cast(scale, (long)q.DataType); // [query_len, num_heads, head_dim] [L,Hq,E]

            var k = GatherKV(AttentionCacheKind.Key, cache, seqId, layerId, queryLen, seqLen, queryStart); // [num_heads, seq_len, head_dim] [H,S,E]
            if (k.Shape[0] != q.Shape[1])
            {
                k = OrtKI.Tile(k, new long[] { q.Shape[1] / k.Shape[0], 1, 1 });
            }

            var attn = OrtKI.Einsum([q, k], "LHE,HSE->HLS"); // [num_heads, query_len, seq_len] [H,L,S]

            // compute causal mask
            var attnBias = OrtKI.Expand(OrtKISharp.Tensor.FromScalar(0f), OrtKISharp.Tensor.MakeTensor([queryLen, seqLen]));
            var tempMask = OrtKISharp.Tensor.MakeTensor(Enumerable.Repeat(1.0f, (int)(queryLen * seqLen)).ToArray(), [queryLen, seqLen]);
            tempMask = OrtKI.Trilu(tempMask, OrtKISharp.Tensor.FromScalar<long>(0), 0);
            attnBias = OrtKI.Where(OrtKI.Equal(tempMask, OrtKISharp.Tensor.FromScalar(1.0f)), attnBias, OrtKI.Expand(OrtKISharp.Tensor.FromScalar(float.NegativeInfinity), OrtKISharp.Tensor.MakeTensor([queryLen, seqLen])));

            attn = attn + OrtKI.Cast(attnBias, (long)attn.DataType);
            attn = OrtKI.Softmax(attn, -1);

            var v = GatherKV(AttentionCacheKind.Value, cache, seqId, layerId, queryLen, seqLen, queryStart); // [num_heads, seq_len, head_dim] [H,S,E]
            if (v.Shape[0] != q.Shape[1])
            {
                v = OrtKI.Tile(v, new long[] { q.Shape[1] / v.Shape[0], 1, 1 });
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
                var block = cache.GetBlock(cacheKind, layerId, headId, blockId);
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

        if (cache.Config.ShardingAxes.Count > 0)
        {
            // var (num_seqs, num_kv_head, head_dim) = (slots.Dimensions[0], slots.Dimensions[1], slots.Dimensions[2]);
            if (cache.Config.AxisPolicies[1].Axes is [1, 2])
            {
                // for xpu
                totalKVHeads *= 2; // recover kv heads.
                for (int did = 0; did < 2; did++)
                {
                    for (int h_id = 0; h_id < cache.Config.NumKVHeads; h_id++)
                    {
                        for (int i = 0; i < numBlocksForSeq; i++)
                        {
                            var blockId = cache.GetBlockId(seqId, i);
                            if (blockId[0] is not -1L)
                            {
                                throw new NotSupportedException();
                            }

                            var tmp_blockId = Tensor.Zeros(blockId.ElementType, blockId.Dimensions);
                            blockId.CopyTo(tmp_blockId);
                            tmp_blockId[0] = did;

                            // 这部分还是可以复用的。
                            var block = cache.GetBlock(cacheKind, layerId, h_id, tmp_blockId);
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
            }
            else if (cache.Config.AxisPolicies[0].Axes is [0] && CompileSessionScope.Current!.CompileOptions.TargetOptions is ICpuTargetOptions { Hierarchies: [[1]] })
            {
                GatherKVCore(cache, (int)numBlocksForSeq, seqId, cacheKind, layerId, (int)seqLen, blockSizeAxis, caches);
            }
            else
            {
                throw new NotSupportedException("topology not support");
            }
        }
        else
        {
            GatherKVCore(cache, (int)numBlocksForSeq, seqId, cacheKind, layerId, (int)seqLen, blockSizeAxis, caches);
        }

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

    private IRType Visit(ITypeInferenceContext context, PagedAttention target, TensorType q, IRType extra)
    {
        return q;
    }

    private IRType Visit(ITypeInferenceContext context, PagedAttention target, DistributedType q, DistributedType extra)
    {
        // for xpu.
        if (q.Placement.Name == "cdxyt")
        {
            if (extra.AxisPolices.All(p => p is SBPBroadCast))
            {
                return new InvalidType("extra should be broadcast!");
            }

            // seq split at x, head split at die and y
            var seqAxis = target.QLayout.IndexOf(AttentionDimKind.Seq);
            var headAxis = target.QLayout.IndexOf(AttentionDimKind.Head);
            var dimAxis = target.QLayout.IndexOf(AttentionDimKind.Dim);
            if (q.AxisPolices[seqAxis] is SBPSplit { Axes: [2] } &&
                q.AxisPolices[headAxis] is SBPSplit { Axes: [1, 3] } &&
                q.AxisPolices[dimAxis] is SBPBroadCast)
            {
                return q;
            }
        }
        else if (q.Placement.Hierarchy.SequenceEqual([1]) && q.AxisPolices.All(x => x is SBPBroadCast))
        {
            return q;
        }

        return new InvalidType("not support distributed type");
    }
}
