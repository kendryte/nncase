// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

public sealed class CreatePagedAttentionKVCacheEvaluator : ITypeInferencer<CreatePagedAttentionKVCache>, ICostEvaluator<CreatePagedAttentionKVCache>, IEvaluator<CreatePagedAttentionKVCache>
{
    public IRType Visit(ITypeInferenceContext context, CreatePagedAttentionKVCache target)
    {
        var num_seqs = context.CheckArgumentType<IRType>(target, CreatePagedAttentionKVCache.NumSeqs);
        var num_tokens = context.CheckArgumentType<IRType>(target, CreatePagedAttentionKVCache.NumTokens);
        var context_lens = context.CheckArgumentType<IRType>(target, CreatePagedAttentionKVCache.ContextLens);
        var seq_lens = context.CheckArgumentType<IRType>(target, CreatePagedAttentionKVCache.SeqLens);
        var block_table = context.CheckArgumentType<IRType>(target, CreatePagedAttentionKVCache.BlockTable);
        var slot_mapping = context.CheckArgumentType<IRType>(target, CreatePagedAttentionKVCache.SlotMapping);
        var num_blocks = context.CheckArgumentType<IRType>(target, CreatePagedAttentionKVCache.NumBlocks);
        var kv_caches = context.CheckArgumentType<IRType>(target, CreatePagedAttentionKVCache.KvCaches);
        return (num_seqs, num_tokens, context_lens, seq_lens, block_table, slot_mapping, num_blocks, kv_caches) switch
        {
            (DistributedType dnum_seqs, DistributedType dnum_tokens, DistributedType dcontext_lens, DistributedType dseq_lens, DistributedType dblock_table, DistributedType dslot_mapping, DistributedType dnum_blocks, DistributedType dkv_caches) => VisitType(context, target, dnum_seqs, dnum_tokens, dcontext_lens, dseq_lens, dblock_table, dslot_mapping, dnum_blocks, dkv_caches),
            (TensorType tnum_seqs, TensorType tnum_tokens, TensorType tcontext_lens, TensorType tseq_lens, TensorType tblock_table, TensorType tslot_mapping, TensorType tnum_blocks, TensorType tkv_caches) => VisitType(context, target, tnum_seqs, tnum_tokens, tcontext_lens, tseq_lens, tblock_table, tslot_mapping, tnum_blocks, tkv_caches),
            _ => new InvalidType("not support type"),
        };
    }

    public Cost Visit(ICostEvaluateContext context, CreatePagedAttentionKVCache target)
    {
        var num_seqs = context.GetArgumentType<IRType>(target, CreatePagedAttentionKVCache.NumSeqs);
        var num_tokens = context.GetArgumentType<IRType>(target, CreatePagedAttentionKVCache.NumTokens);
        var context_lens = context.GetArgumentType<IRType>(target, CreatePagedAttentionKVCache.ContextLens);
        var seq_lens = context.GetArgumentType<IRType>(target, CreatePagedAttentionKVCache.SeqLens);
        var block_table = context.GetArgumentType<IRType>(target, CreatePagedAttentionKVCache.BlockTable);
        var slot_mapping = context.GetArgumentType<IRType>(target, CreatePagedAttentionKVCache.SlotMapping);
        var num_blocks = context.GetArgumentType<IRType>(target, CreatePagedAttentionKVCache.NumBlocks);
        var kv_caches = context.GetArgumentType<IRType>(target, CreatePagedAttentionKVCache.KvCaches);

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(num_seqs) + CostUtility.GetMemoryAccess(num_tokens) + CostUtility.GetMemoryAccess(context_lens) + CostUtility.GetMemoryAccess(seq_lens) + CostUtility.GetMemoryAccess(block_table) + CostUtility.GetMemoryAccess(slot_mapping) + CostUtility.GetMemoryAccess(num_blocks) + CostUtility.GetMemoryAccess(kv_caches),
        };
    }

    public IValue Visit(IEvaluateContext context, CreatePagedAttentionKVCache target)
    {
        var num_seqs = context.GetArgumentValueAsScalar<int>(target, CreatePagedAttentionKVCache.NumSeqs);
        var num_tokens = context.GetArgumentValueAsScalar<int>(target, CreatePagedAttentionKVCache.NumTokens);
        var context_lens = context.GetArgumentValueAsTensor<long>(target, CreatePagedAttentionKVCache.ContextLens);
        var seq_lens = context.GetArgumentValueAsTensor<long>(target, CreatePagedAttentionKVCache.SeqLens);
        var block_table = context.GetArgumentValueAsTensor<long>(target, CreatePagedAttentionKVCache.BlockTable);
        var slot_mapping = context.GetArgumentValueAsTensor<long>(target, CreatePagedAttentionKVCache.SlotMapping);
        var num_blocks = context.GetArgumentValueAsScalar<int>(target, CreatePagedAttentionKVCache.NumBlocks);
        var kv_caches = context.GetArgumentValueAsTensor(target, CreatePagedAttentionKVCache.KvCaches);

        var kv_cache = new RefPagedAttentionKVCache(target.Config, num_seqs, num_tokens, context_lens, seq_lens, block_table, slot_mapping, num_blocks, kv_caches);
        return Value.FromTensor(Tensor.FromScalar(new Reference<IPagedAttentionKVCache>(kv_cache)));
    }

    private IRType CheckAllBroadCast(DistributedType distributedType, [System.Runtime.CompilerServices.CallerArgumentExpression("distributedType")] string? name = null)
    {
        if (!distributedType.AxisPolicies.All(x => x is SBPBroadCast))
        {
            return new InvalidType($"{name} is not all broadcast");
        }

        return distributedType;
    }

    private IRType VisitType(ITypeInferenceContext context, CreatePagedAttentionKVCache target, DistributedType num_seqs, DistributedType num_tokens, DistributedType context_lens, DistributedType seq_lens, DistributedType block_table, DistributedType slot_mapping, DistributedType num_blocks, DistributedType kv_caches)
    {
        var validType = VisitType(context, target, num_seqs.TensorType, num_tokens.TensorType, context_lens.TensorType, seq_lens.TensorType, block_table.TensorType, slot_mapping.TensorType, num_blocks.TensorType, kv_caches.TensorType);
        if (validType is InvalidType)
        {
            return validType;
        }

        if (CheckAllBroadCast(num_tokens) is InvalidType iv)
        {
            return iv;
        }

        if (CheckAllBroadCast(context_lens) is InvalidType iv1)
        {
            return iv1;
        }

        if (CheckAllBroadCast(seq_lens) is InvalidType iv2)
        {
            return iv2;
        }

        if (CheckAllBroadCast(block_table) is InvalidType iv3)
        {
            return iv3;
        }

        if (CheckAllBroadCast(slot_mapping) is InvalidType iv4)
        {
            return iv4;
        }

        if (CheckAllBroadCast(num_blocks) is InvalidType iv5)
        {
            return iv5;
        }

        if (kv_caches.Placement.Name == "cdxyt")
        {
            if (kv_caches.AxisPolicies[0] is SBPSplit { Axes: [1] } &&
                kv_caches.AxisPolicies[1] is SBPSplit { Axes: [2, 3] } &&
                kv_caches.AxisPolicies.Skip(2).All(x => x is SBPBroadCast))
            {
                return validType;
            }
        }

        return new InvalidType("not support distributed kv caches");
    }

    private IRType VisitType(ITypeInferenceContext context, CreatePagedAttentionKVCache target, TensorType num_seqs, TensorType num_tokens, TensorType context_lens, TensorType seq_lens, TensorType block_table, TensorType slot_mapping, TensorType num_blocks, TensorType kv_caches)
    {
        if (!num_seqs.IsScalar)
        {
            return new InvalidType("num_seqs is not scalar");
        }

        if (!num_tokens.IsScalar)
        {
            return new InvalidType("num_tokens is not scalar");
        }

        if (context_lens.Shape.Rank != 1)
        {
            return new InvalidType("context_lens rank != 1");
        }

        if (seq_lens.Shape.Rank != 1)
        {
            return new InvalidType("seq_lens rank != 1");
        }

        if (block_table.Shape.Rank != 3)
        {
            return new InvalidType("block_table rank != 3");
        }

        if (slot_mapping.Shape.Rank != 2)
        {
            return new InvalidType("slot_mapping rank != 2");
        }

        if (!num_blocks.IsScalar)
        {
            return new InvalidType("slot_mapping rank != 2");
        }

        if (!num_blocks.IsScalar)
        {
            return new InvalidType("slot_mapping rank != 2");
        }

        if (kv_caches.Shape.Rank < target.Config.CacheLayout.Count)
        {
            return new InvalidType("kv_caches shape < CacheLayout.Count");
        }

        return new ReferenceType(new PagedAttentionKVCacheType()
        {
            Config = target.Config,
        });
    }
}
