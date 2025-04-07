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
    public static OrtKISharp.Tensor RefPagedAttn(OrtKISharp.Tensor query, PagedAttentionKVCache cache, float scale, long layerId)
    {
        long numSeqs = cache.SeqLens.Length;

        var outputs = new List<OrtKISharp.Tensor>();
        long startIdx = 0;

        for (int i = 0; i < numSeqs; i++)
        {
            long queryLen = cache.SeqLens[i] - cache.ContextLens[i];
            long seqLen = cache.SeqLens[i];
            var q = OrtKI.Slice(query, OrtKISharp.Tensor.MakeTensor([startIdx], [1]), OrtKISharp.Tensor.MakeTensor([startIdx + queryLen], [1]), OrtKISharp.Tensor.MakeTensor([0L], [1]), OrtKISharp.Tensor.MakeTensor([1L], [1]));
            q = q * scale;

            long numBlocks = (seqLen + cache.BlockSize - 1) / cache.BlockSize;
            var blockIndices = Enumerable.Range(0, (int)numBlocks).Select(j => cache.BlockTables[i, j]).ToArray();

            var kcaches = cache.KCaches.ToOrtTensor();
            kcaches.Reshape([cache.NumLayers, cache.NumBlocks * cache.BlockSize, cache.NumKVHeads, cache.HeadSize]);
            var layer_kcaches = OrtKI.Gather(kcaches, layerId, 0);
            var k = OrtKI.Slice(layer_kcaches, OrtKISharp.Tensor.MakeTensor([0L], [1]), OrtKISharp.Tensor.MakeTensor([seqLen], [1]), OrtKISharp.Tensor.MakeTensor([0L], [1]), OrtKISharp.Tensor.MakeTensor([1L], [1]));

            var vcaches = cache.VCaches.ToOrtTensor();
            vcaches.Reshape([cache.NumLayers, cache.NumBlocks * cache.BlockSize, cache.NumKVHeads, cache.HeadSize]);
            var layer_vcaches = OrtKI.Gather(vcaches, layerId, 0);

            var v = OrtKI.Slice(layer_vcaches, OrtKISharp.Tensor.MakeTensor([0L], [1]), OrtKISharp.Tensor.MakeTensor([seqLen], [1]), OrtKISharp.Tensor.MakeTensor([0L], [1]), OrtKISharp.Tensor.MakeTensor([1L], [1]));

            if (q.Shape[1] != k.Shape[1])
            {
                throw new NotSupportedException($"not support {q.Shape[1]} {k.Shape[1]}");
            }

            var attn = OrtKI.Einsum([q, k], "qhd,khd->hqk");

            // compute causal mask
            var emptyMask = Tensor.FromScalar(1.0f, [queryLen, seqLen]).ToOrtTensor();
            var maskCond = OrtKI.Equal(OrtKI.Trilu(emptyMask, queryLen - seqLen, 0), 1.0f);
            var attnBias = OrtKI.Expand(OrtKISharp.Tensor.FromScalar(0f), OrtKISharp.Tensor.MakeTensor([queryLen, seqLen]));
            var neginf = OrtKI.Expand(OrtKISharp.Tensor.FromScalar(float.NegativeInfinity), OrtKISharp.Tensor.MakeTensor([queryLen, seqLen]));
            attnBias = OrtKI.Where(maskCond, attnBias, neginf);
            attn = attn + attnBias;
            attn = OrtKI.Softmax(attn, -1);
            var outTensor = OrtKI.Einsum([attn, v], "hqk,khd->qhd");

            outputs.Add(outTensor);
            startIdx += queryLen;
        }

        return OrtKI.Concat(outputs.ToArray(), 0L);
    }

    public static void CacheFlash(Tensor<float> key, Tensor<float> value, PagedAttentionKVCache cache, long layerId)
    {
        var num_tokens = cache.SlotMaping.Length;
        var key_span = key.Buffer.Span;
        var val_span = value.Buffer.Span;
        var key_cache_span = cache.KCaches.Buffer.Span;
        var val_cache_span = cache.VCaches.Buffer.Span;
        for (int i = 0; i < num_tokens; i++)
        {
            var unit_length = cache.NumKVHeads * cache.HeadSize;
            var token_key_span = key_span.Slice(i * unit_length, unit_length);
            var token_val_span = val_span.Slice(i * unit_length, unit_length);

            var slot = cache.SlotMaping[i];
            var token_key_cache_span = key_cache_span.Slice((int)TensorUtilities.GetIndex(cache.KCaches.Strides, [layerId, slot / cache.BlockSize, slot % cache.BlockSize, 0, 0]), unit_length);
            var token_val_cache_span = val_cache_span.Slice((int)TensorUtilities.GetIndex(cache.KCaches.Strides, [layerId, slot / cache.BlockSize, slot % cache.BlockSize, 0, 0]), unit_length);

            token_key_span.CopyTo(token_key_cache_span);
            token_val_span.CopyTo(token_val_cache_span);
        }
    }

    public IRType Visit(ITypeInferenceContext context, PagedAttention target)
    {
        var q = context.CheckArgumentType<IRType>(target, PagedAttention.Q);
        var k = context.CheckArgumentType<IRType>(target, PagedAttention.K);
        var v = context.CheckArgumentType<IRType>(target, PagedAttention.V);

        return (q, k, v) switch
        {
            (DistributedType dq, DistributedType dk, DistributedType dv) => Visit(context, target, dq, dk, dv),
            (TensorType tq, TensorType tk, TensorType tv) => Visit(context, target, tq, tk, tv),
            _ => new InvalidType("not support type"),
        };
    }

    public Cost Visit(ICostEvaluateContext context, PagedAttention target)
    {
        var qType = context.GetArgumentType<IRType>(target, PagedAttention.Q);
        var kType = context.GetArgumentType<IRType>(target, PagedAttention.K);
        var vType = context.GetArgumentType<IRType>(target, PagedAttention.V);
        var returnType = context.GetReturnType<IRType>();

        // cycles = softmax((q @ k^t) + mask) @ v.
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(qType) + CostUtility.GetMemoryAccess(vType) + CostUtility.GetMemoryAccess(kType),

            // todo [CostFactorNames.CPUCycles].
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public IValue Visit(IEvaluateContext context, PagedAttention target)
    {
        var q = context.GetOrtArgumentValue(target, PagedAttention.Q);
        var k = context.GetArgumentValueAsTensor<float>(target, PagedAttention.K);
        var v = context.GetArgumentValueAsTensor<float>(target, PagedAttention.V);
        var kvcache = context.GetArgumentValue(target, PagedAttention.KVCache).AsObjectRef<PagedAttentionKVCache>();
        CacheFlash(k, v, kvcache, target.LayerId);
        return RefPagedAttn(q, kvcache, 1.0f, target.LayerId).ToValue();
    }

    private IRType Visit(ITypeInferenceContext context, PagedAttention target, TensorType q, TensorType k, TensorType v)
    {
        var dims = new Dimension[q.Shape.Rank];
        for (int i = 0; i < q.Shape.Rank - 1; i++)
        {
            dims[i] = q.Shape[i];
        }

        dims[^1] = v.Shape[^1];

        return new TensorType(q.DType, dims);
    }

    private IRType Visit(ITypeInferenceContext context, PagedAttention target, DistributedType q, DistributedType k, DistributedType v)
    {
        return new InvalidType("not support distributed type");
    }
}
