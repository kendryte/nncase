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
        for (int requestId = 0; requestId < cache.NumRequests; requestId++)
        {
            var seqLen = cache.GetSeqLens(requestId);
            var queryLen = seqLen - cache.GetContextLength(requestId);
            var q = OrtKI.Slice(query, OrtKISharp.Tensor.MakeTensor([queryStart], [1]), OrtKISharp.Tensor.MakeTensor([queryStart + queryLen], [1]), OrtKISharp.Tensor.MakeTensor([0L], [1]), OrtKISharp.Tensor.MakeTensor([1L], [1]));
            q = q * scale;

            var k = GatherKV(AttentionCacheKind.Key, cache, requestId, layerId, queryLen, seqLen, queryStart); // [num_heads, seq_len, head_dim]
            var attn = OrtKI.Einsum([q, k], "qhd,hkd->qhk"); // [query_len, num_heads, seq_len]

            // compute causal mask
            var emptyMask = Tensor.FromScalar(1.0f, [queryLen, seqLen]).ToOrtTensor();
            var maskCond = OrtKI.Equal(OrtKI.Trilu(emptyMask, queryLen - seqLen, 0), 1.0f);
            var attnBias = OrtKI.Expand(OrtKISharp.Tensor.FromScalar(0f), OrtKISharp.Tensor.MakeTensor([queryLen, seqLen]));
            var neginf = OrtKI.Expand(OrtKISharp.Tensor.FromScalar(float.NegativeInfinity), OrtKISharp.Tensor.MakeTensor([queryLen, seqLen]));
            attnBias = OrtKI.Where(maskCond, attnBias, neginf);
            attn = attn + attnBias;
            attn = OrtKI.Softmax(attn, -1);

            var v = GatherKV(AttentionCacheKind.Value, cache, requestId, layerId, queryLen, seqLen, queryStart); // [num_heads, seq_len, head_dim]
            var outTensor = OrtKI.Einsum([attn, v], "qhv,hvd->qhd"); // [query_len, num_heads, head_dim]

            outputs.Add(outTensor);
            queryStart += queryLen;
        }

        return OrtKI.Concat(outputs.ToArray(), 0L);
    }

    private static OrtKISharp.Tensor GatherKV(AttentionCacheKind cacheKind, IPagedAttentionKVCache cache, int requestId, int layerId, long queryLen, long seqLen, long queryStart)
    {
        var caches = new List<OrtKISharp.Tensor>();
        var contextBlockIds = cache.GetContextBlockIds(requestId);
        var outputSlotIds = cache.GetOutputSlotIds();

        // context seqs
        long slotsRead = 0;
        var blocksCount = contextBlockIds.Dimensions[0];
        for (int i = 0; i < blocksCount; i++)
        {
            var blockId = contextBlockIds[i];
            var block = cache.GetBlock(cacheKind, layerId, blockId);
            var slotsCount = (int)System.Math.Min(seqLen - slotsRead, cache.Config.BlockSize);
            var slots = cache.GetSlots(block, 0, slotsCount);
            caches.Add(slots.ToOrtTensor());
            slotsRead += slotsCount;
        }

        // output slots
        for (int i = 0; i < queryLen; i++)
        {
            var slotId = outputSlotIds[queryStart + i];
            var slot = cache.GetSlot(cacheKind, layerId, slotId);
            caches.Add(slot.ToOrtTensor());
        }

        return OrtKI.Concat(caches.ToArray(), 1L);
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
