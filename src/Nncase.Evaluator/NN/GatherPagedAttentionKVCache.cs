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

public sealed class GatherPagedAttentionKVCacheEvaluator : ITypeInferencer<GatherPagedAttentionKVCache>, ICostEvaluator<GatherPagedAttentionKVCache>, IEvaluator<GatherPagedAttentionKVCache>
{
    public IRType Visit(ITypeInferenceContext context, GatherPagedAttentionKVCache target)
    {
        var shardId = context.CheckArgumentType<IRType>(target, GatherPagedAttentionKVCache.ShardId);
        var kvCache = context.CheckArgumentType<IRType>(target, GatherPagedAttentionKVCache.KVCaches);
        return shardId switch
        {
            DistributedType dslots => Visit(context, target, dslots, kvCache),
            TensorType tslots => Visit(context, target, tslots, kvCache),
            _ => new InvalidType("not support type"),
        };
    }

    public Cost Visit(ICostEvaluateContext context, GatherPagedAttentionKVCache target)
    {
        var slotsType = context.GetArgumentType<IRType>(target, GatherPagedAttentionKVCache.ShardId);
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(slotsType) + CostUtility.GetMemoryAccess(returnType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(slotsType) + CostUtility.GetMemoryAccess(returnType),
        };
    }

    public IValue Visit(IEvaluateContext context, GatherPagedAttentionKVCache target)
    {
        var kvCaches = context.GetArgumentValue(target, GatherPagedAttentionKVCache.KVCaches);
        return Value.FromTensor(GatherCache(kvCaches.AsTensor().Cast<Reference<IPagedAttentionKVCache>>()));
    }

    private static Tensor GatherCache(Tensor<Reference<IPagedAttentionKVCache>> kvCaches)
    {
        var refernece = kvCaches.Single();
        var kvcahe = (RefPagedAttentionKVCache)refernece.Value;
        var ortKVCacheTensor = kvcahe.KVCaches.ToOrtTensor();
        var config = (IPagedAttentionConfig)kvcahe.Config;
        var lanesShape = ortKVCacheTensor.Shape.Skip(config.ShardingAxes.Count + config.CacheLayout.Count).ToArray();

        var perms = Enumerable.Range(0, ortKVCacheTensor.Shape.Length).Select(i => new List<long> { i }).ToList();
        for (int i = 0; i < config.ShardingAxes.Count; i++)
        {
            var axis = config.CacheLayout.IndexOf(config.ShardingAxes[i]);
            perms[config.ShardingAxes.Count + axis].Insert(0, perms[i][0]);
            perms[i].RemoveAt(0);
        }

        var perm = perms.SelectMany(x => x).ToArray();
        var transKVCacheTensor = OrtKI.Transpose(ortKVCacheTensor, perm);
        var logicalTensorType = config.GetLogicalTensorType(kvcahe.NumBlocks);
        var logicalShape = logicalTensorType.Shape.ToValueArray();
        var shape = logicalShape.Concat(lanesShape).ToArray();
        var reshaped = OrtKI.Reshape(transKVCacheTensor, shape, 0);
        var tensor = reshaped.ToTensor(logicalTensorType);
        return tensor;
    }

    private IRType Visit(ITypeInferenceContext context, GatherPagedAttentionKVCache target, TensorType shardId, IRType kvCache)
    {
        if (shardId.DType != DataTypes.Int64)
        {
            return new InvalidType("only support int64!");
        }

        if (kvCache is not TensorType { DType: ReferenceType { ElemType: PagedAttentionKVCacheType pagedAttentionKVCacheType } })
        {
            return new InvalidType("only support PagedAttentionKVCache!");
        }

        return pagedAttentionKVCacheType.Config.GetLogicalTensorType(target.NumBlocks);
    }

    private IRType Visit(ITypeInferenceContext context, GatherPagedAttentionKVCache target, DistributedType shardId, IRType kvCache)
    {
        if (Visit(context, target, shardId.TensorType, kvCache) is InvalidType invalidType)
        {
            return invalidType;
        }

        if (!shardId.AxisPolices.All(sbp => sbp is SBPBroadCast))
        {
            return new InvalidType("only support broadcast!");
        }

        if (kvCache is not TensorType { DType: ReferenceType { ElemType: PagedAttentionKVCacheType pagedAttentionKVCacheType } })
        {
            return new InvalidType("only support PagedAttentionKVCache!");
        }

        var tensorType = pagedAttentionKVCacheType.Config.GetLogicalTensorType(target.NumBlocks);
        var axisPolices = Enumerable.Repeat<SBP>(SBP.B, tensorType.Shape.Rank).ToArray();
        for (int i = 0; i < pagedAttentionKVCacheType.Config.ShardingAxes.Count; i++)
        {
            var dimName = pagedAttentionKVCacheType.Config.ShardingAxes[i];
            axisPolices[pagedAttentionKVCacheType.Config.CacheLayout.IndexOf(dimName)] = pagedAttentionKVCacheType.Config.AxisPolicies[i];
        }

        return new DistributedType(tensorType, axisPolices.ToArray(), shardId.Placement);
    }
}
