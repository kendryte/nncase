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
        return q switch
        {
            DistributedType dq => Visit(context, target, dq),
            TensorType tq => Visit(context, target, tq),
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
        return RefPagedAttn(q, kvCaches, 1.0f, target.LayerId).ToValue();
    }

    private static OrtKISharp.Tensor RefPagedAttn(OrtKISharp.Tensor query, Tensor<Reference<IPagedAttentionKVCache>> kvCaches, float scale, int layerId)
    {
        // TODO: Support DP
        if (kvCaches.Length != 1)
        {
            throw new ArgumentException($"kvCaches length {kvCaches.Length} != 1");
        }

        var cache = kvCaches.Single().Value;
        var outputs = new List<OrtKISharp.Tensor>();
        long queryStart = 0;
        for (int seqId = 0; seqId < cache.NumSeqs; seqId++)
        {
            var seqLen = cache.GetSeqLen(seqId);
            var queryLen = seqLen - cache.GetContextLen(seqId);
            var q = OrtKI.Slice(query, new long[] { queryStart }, new long[] { queryStart + queryLen }, new long[] { 0L }, new long[] { 1L });

            // unpack for q
            if (cache.Config.PackedAxes.Contains(PagedAttentionDimKind.HeadDim))
            {
                q = q.Unpack(2);
            }

            if (cache.Config.PackedAxes.Contains(PagedAttentionDimKind.NumKVHeads))
            {
                q = q.Unpack(1);
            }

            q = q * scale; // [query_len, num_heads, head_dim] [L,Hq,E]

            var k = GatherKV(AttentionCacheKind.Key, cache, seqId, layerId, queryLen, seqLen, queryStart); // [num_heads, seq_len, head_dim] [H,S,E]
            var attn = OrtKI.Einsum([q, k], "LHE,HSE->HLS"); // [num_heads, query_len, seq_len] [H,L,S]

            // compute causal mask
            var attnBias = OrtKI.Expand(OrtKISharp.Tensor.FromScalar(0f), OrtKISharp.Tensor.MakeTensor([queryLen, seqLen]));
            var tempMask = OrtKISharp.Tensor.MakeTensor(Enumerable.Repeat(1.0f, (int)(queryLen * seqLen)).ToArray(), [queryLen, seqLen]);
            tempMask = OrtKI.Trilu(tempMask, OrtKISharp.Tensor.FromScalar<long>(0), 0);
            attnBias = OrtKI.Where(OrtKI.Equal(tempMask, OrtKISharp.Tensor.FromScalar(1.0f)), attnBias, OrtKI.Expand(OrtKISharp.Tensor.FromScalar(float.NegativeInfinity), OrtKISharp.Tensor.MakeTensor([queryLen, seqLen])));

            attn = attn + attnBias;
            attn = OrtKI.Softmax(attn, -1);

            var v = GatherKV(AttentionCacheKind.Value, cache, seqId, layerId, queryLen, seqLen, queryStart); // [num_heads, seq_len, head_dim] [H,S,E]
            var output = OrtKI.Einsum([attn, v], "HLS,HSE->LHE"); // [query_len, num_heads, head_dim] [L,H,E]

            // repack for q
            if (cache.Config.PackedAxes.Contains(PagedAttentionDimKind.NumKVHeads))
            {
                output = output.Pack(cache.Config.Lanes[cache.Config.PackedAxes.IndexOf(PagedAttentionDimKind.NumKVHeads)], 1);
            }

            if (cache.Config.PackedAxes.Contains(PagedAttentionDimKind.HeadDim))
            {
                output = output.Pack(cache.Config.Lanes[cache.Config.PackedAxes.IndexOf(PagedAttentionDimKind.HeadDim)], 2);
            }

            outputs.Add(output);
            queryStart += queryLen;
        }

        return OrtKI.Concat(outputs.ToArray(), 0L);
    }

    private static OrtKISharp.Tensor GatherKV(AttentionCacheKind cacheKind, IPagedAttentionKVCache cache, int seqId, int layerId, long queryLen, long seqLen, long queryStart)
    {
        var caches = new List<OrtKISharp.Tensor>();
        var blockIds = cache.GetBlockIds(seqId);
        var numBlocksForSeq = MathUtility.CeilDiv(cache.GetSeqLen(seqId), cache.Config.BlockSize);

        // block layout is construct from `head dim, block_size`. but we don't know the concrete shape.
        var blockLayout = cache.Config.BlockLayout;
        var blockSizeAxis = blockLayout.IndexOf(PagedAttentionDimKind.BlockSize);
        if (blockSizeAxis < 0)
        {
            throw new InvalidOperationException("block layout not contain block size");
        }

        var headDimAxis = blockLayout.IndexOf(PagedAttentionDimKind.HeadDim);
        if (headDimAxis < 0)
        {
            throw new InvalidOperationException("block layout not contain head dim");
        }

        for (int headId = 0; headId < cache.Config.NumKVHeads; headId++)
        {
            for (int i = 0; i < numBlocksForSeq; i++)
            {
                var blockId = blockIds[i];
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

        // caches is (head * blocks) * [BlockLayout + lanes], concat at block size axis.
        var concatCache = OrtKI.Concat(caches.ToArray(), blockSizeAxis);

        // unpack head dim
        if (cache.Config.PackedAxes.Contains(PagedAttentionDimKind.HeadDim))
        {
            concatCache = concatCache.Unpack(headDimAxis);
        }

        // transpose to [num_head * seq_len, head_dim]
        if (blockLayout is [PagedAttentionDimKind.HeadDim, PagedAttentionDimKind.BlockSize])
        {
            concatCache = OrtKI.Transpose(concatCache, [1, 0]);
        }

        var newShape = concatCache.Shape.ToList(); // split head and seq len dim.
        newShape.Insert(0, cache.Config.NumKVHeads);
        newShape[1] /= cache.Config.NumKVHeads;
        concatCache.Reshape(newShape.ToArray());
        return concatCache; // [num_heads, seq_len, head_dim]
    }

    private IRType Visit(ITypeInferenceContext context, PagedAttention target, TensorType q)
    {
        var headDim = q.Shape[^1];
        var dims = q.Shape.ToArray();
        dims[^1] = headDim;
        return q with { Shape = dims };
    }

    private IRType Visit(ITypeInferenceContext context, PagedAttention target, DistributedType q)
    {
        return new InvalidType("not support distributed type");
    }
}
