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

public abstract record RefAttentionKVCache(
    IAttentionConfig Config,
    int NumSeqs,
    int NumTokens,
    Tensor<long> ContextLens,
    Tensor<long> SeqLens) : IAttentionKVCache
{
    public long ContextLen(int requestId) => ContextLens[requestId];

    public long SeqLen(int requestId) => SeqLens[requestId];
}

[JsonConverter(typeof(PagedAttentionKVCacheJsonConverterFactory))]
public sealed record RefPagedAttentionKVCache(
    IAttentionConfig Config,
    int NumSeqs,
    int NumTokens,
    Tensor<long> ContextLens,
    Tensor<long> SeqLens,
    Tensor<long> BlockTable,
    Tensor<long> SlotMapping,
    int NumBlocks,
    Tensor KVCaches)
    : RefAttentionKVCache(
        Config,
        NumSeqs,
        NumTokens,
        ContextLens,
        SeqLens), IPagedAttentionKVCache
{
    IPagedAttentionConfig IPagedAttentionKVCache.Config => (IPagedAttentionConfig)Config;

    private IPagedAttentionConfig PagedAttentionConfig => (IPagedAttentionConfig)Config;

    public Tensor GetBlockId(int seqId, int contextId)
    {
        Debug.Assert(BlockTable.Rank == 3, "BlockTable must be 3D.");
        return BlockTable.View([seqId, contextId, 0], [1, 1, BlockTable.Dimensions[2]]).Squeeze(0, 1).AsContiguous();
    }

    public Tensor GetSlotId(int tokenId)
    {
        Debug.Assert(SlotMapping.Rank == 2, "SlotMapping must be 2D.");
        return SlotMapping.View([tokenId, 0], [1, SlotMapping.Dimensions[1]]).Squeeze(0).AsContiguous();
    }

    public Tensor GetBlock(AttentionCacheKind kind, int layerId, int headId, Tensor blockId)
    {
        var blockView = GetBlockViewFromStorage(kind, layerId, headId, blockId);
        return blockView.AsContiguous();
    }

    public void UpdateBlock(AttentionCacheKind kind, int layerId, int headId, Tensor blockId, Tensor block)
    {
        block.CopyTo(GetBlockViewFromStorage(kind, layerId, headId, blockId));
    }

    public Tensor GetSlot(AttentionCacheKind kind, int layerId, int headId, Tensor slotId)
    {
        return GetSlotViewFromStorage(kind, layerId, headId, slotId);
    }

    public void UpdateSlot(AttentionCacheKind kind, int layerId, int headId, Tensor slotId, Tensor slot)
    {
        var destView = GetSlotViewFromStorage(kind, layerId, headId, slotId);
        slot.CopyTo(destView);
    }

    public void UpdateSlots(AttentionCacheKind kind, int layerId, int headId, Tensor slots)
    {
        // slots : [num_tokens, numHeads, headDim]
        for (int i = 0; i < NumTokens; i++)
        {
            var slotId = GetSlotId(i);
            var slot = slots.View([i, headId, 0], [1, 1, slots.Dimensions[2]]).Squeeze([0, 1]);
            UpdateSlot(kind, layerId, headId, slotId, slot);
        }
    }

    public long[] LogicalCacheDimensions()
    {
        return KVCaches.Dimensions.ToArray();
    }

    /// <summary>
    /// Materializes the slot mapping ID.
    /// </summary>
    /// <param name="slotMappingTensor"> physical slot mapping tensor. </param>
    /// <param name="indices"> the indices of the slot mapping tensor. </param>
    /// <param name="logicalSlotId"> the logical slot ID. </param>
    /// <param name="numBlocks"> the number of blocks. </param>
    /// <param name="placement"> the placement information. </param>
    /// <param name="config"> the attention configuration. </param>
    public static void MaterializeSlotMappingId(Tensor<long> slotMappingTensor, long[] indices, long logicalSlotId, int numBlocks, Placement placement, IPagedAttentionConfig config)
    {
        var physicalSlotId = logicalSlotId;

        for (int shardId = 0; shardId < config.ShardingAxes.Count; shardId++)
        {
            switch (config.ShardingAxes[shardId])
            {
                case PagedKVCacheDimKind.NumBlocks:
                    var parallelism = config.AxisPolicies[shardId].Axes.Select(axis => placement.Hierarchy[axis]).Product();
                    if (numBlocks < parallelism && !DistributedUtility.IsDivideExactly(numBlocks, parallelism))
                    {
                        throw new InvalidOperationException("numBlocks < parallelism");
                    }

                    var numBlockTile = numBlocks / parallelism * config.BlockSize;
                    var value = (int)System.Math.DivRem(physicalSlotId, numBlockTile, out physicalSlotId);
                    slotMappingTensor[indices] = value;
                    break;
                case PagedKVCacheDimKind.NumKVHeads when config.AxisPolicies[shardId].Axes.Count == 1:
                    slotMappingTensor[indices] = -1; // todo should matching the kv sharding.
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(config), $"ShardingAxes: {config.ShardingAxes[shardId]}, AxisPolicy: {config.AxisPolicies[shardId]}, Placement: {placement}");
            }

            indices[^1]++;
        }

        slotMappingTensor[indices] = physicalSlotId;
    }

    public static void MaterializeBlockTable(Tensor<long> blockTableTensor, long[] indices, long logicalBlockId, int numBlocks, Placement placement, IPagedAttentionConfig config)
    {
        long physicalBlockId = logicalBlockId;
        for (int topoId = 0; topoId < config.ShardingAxes.Count; topoId++)
        {
            switch (config.ShardingAxes[topoId])
            {
                case PagedKVCacheDimKind.NumBlocks:
                    var parallelism = config.AxisPolicies[topoId].Axes.Select(axis => placement.Hierarchy[axis]).Product();
                    if (numBlocks < parallelism && !DistributedUtility.IsDivideExactly(numBlocks, parallelism))
                    {
                        throw new InvalidOperationException("numBlocks < parallelism");
                    }

                    var numBlockTile = numBlocks / parallelism;
                    var value = (int)System.Math.DivRem(physicalBlockId, numBlockTile, out physicalBlockId);
                    blockTableTensor[indices] = value;
                    break;
                case PagedKVCacheDimKind.NumKVHeads:
                    blockTableTensor[indices] = -1;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(config));
            }

            indices[^1]++;
        }

        blockTableTensor[indices] = physicalBlockId;
    }

    public static (int HeadId, Tensor<long> SlotId) PhysicalizeSlotMappingId(Tensor<long> slotId, int headId, int numBlocks, Placement placement, IPagedAttentionConfig config)
    {
        var headIdCopy = headId;
        var slotIdCopy = slotId.AsContiguous(true);
        var cacheDimensions = config.GetLogicalShardTensorType(numBlocks, placement).Shape.ToValueArray();
        for (int shardId = 0; shardId < config.ShardingAxes.Count; shardId++)
        {
            switch (config.ShardingAxes[shardId])
            {
                case PagedKVCacheDimKind.NumKVHeads when slotIdCopy[shardId] is -1L:
                    var headTile = config.NumKVHeads / (int)cacheDimensions[shardId];
                    slotIdCopy[shardId] = System.Math.DivRem(headIdCopy, headTile, out headIdCopy);
                    break;
                case PagedKVCacheDimKind.NumBlocks when slotIdCopy[shardId] is not -1L:
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(slotId));
            }
        }

        return (headIdCopy, slotIdCopy);
    }

    /// <summary>
    /// all vars are [seq,head,dim] layout.
    /// </summary>
    public static (Expr Root, Var QueryVar, List<Var[]> KVVars, Var KVCacheObjVar) BuildPagedAttentionKernel(long[] queryLens, long[] seqLens, int numQHeads, int numBlocks, AttentionDimKind[] qLayout, AttentionDimKind[] kvLayout, IPagedAttentionConfig config, BuildKernelOptions options)
    {
        var numTokens = queryLens.Sum();
        var numTokensVar = new IR.DimVar("num_tokens") { Metadata = { Range = new(1.0f, options.DynamicMaxTokens) } };
        Shape defaultQDimenions = options.DynamicShape ? new RankedShape(numTokensVar, numQHeads, config.HeadDim) : new long[] { numTokens, numQHeads, config.HeadDim };
        var queryVar = new Var("query", new TensorType(config.KVPrimType, defaultQDimenions));
        var kvVars = new List<Var[]>();
        var kvCacheObjVar = new Var("kvCache", TensorType.Scalar(
            new ReferenceType(new PagedAttentionKVCacheType { Config = config })));

        // Create vars for each layer
        Shape defaultKDimenions = options.DynamicShape ? new RankedShape(numTokensVar, config.NumKVHeads, config.HeadDim) : new long[] { numTokens, config.NumKVHeads, config.HeadDim };
        for (int layerId = 0; layerId < config.NumLayers; layerId++)
        {
            var keyVar = new Var($"key_{layerId}", new TensorType(config.KVPrimType, defaultKDimenions));
            var valueVar = new Var($"value_{layerId}", new TensorType(config.KVPrimType, defaultKDimenions));
            kvVars.Add([keyVar, valueVar]);
        }

        // Build computation graph
        var (q_lanes, q_packed_axes) = GetQKVPackParams(config, qLayout);
        var (kv_lanes, kv_packed_axes) = GetQKVPackParams(config, kvLayout);
        var padedQuery = options.DynamicShape ? IR.F.NN.Pad(queryVar, new IR.Shapes.Paddings(new(0, options.DynamicMaxTokens - numTokensVar), new(0, 0), new(0, 0)), PadMode.Constant, Tensor.Zeros(config.KVPrimType, [])) : (Expr)queryVar;
        var transedQuery = IR.F.Tensors.Transpose(padedQuery, qLayout.Select(x => (int)x).ToArray());
        var packedQuery = q_lanes.Length > 0 ? IR.F.Tensors.Pack(transedQuery, q_lanes, q_packed_axes) : transedQuery;
        Expr updatedKVCache = None.Default;
        for (int layerId = 0; layerId < config.NumLayers; layerId++)
        {
            var (keyVar, valueVar) = (kvVars[layerId][0], kvVars[layerId][1]);

            var padedKey = options.DynamicShape ? IR.F.NN.Pad(keyVar, new IR.Shapes.Paddings(new(0, options.DynamicMaxTokens - numTokensVar), new(0, 0), new(0, 0)), PadMode.Constant, Tensor.Zeros(config.KVPrimType, [])) : (Expr)keyVar;
            var transedKey = IR.F.Tensors.Transpose(padedKey, kvLayout.Select(x => (int)x).ToArray());
            var packedKey = kv_lanes.Length > 0 ? IR.F.Tensors.Pack(transedKey, kv_lanes, kv_packed_axes) : transedKey;
            updatedKVCache = IR.F.NN.UpdatePagedAttentionKVCache(
                packedKey,
                kvCacheObjVar,
                AttentionCacheKind.Key,
                layerId,
                kvLayout);

            var padedValue = options.DynamicShape ? IR.F.NN.Pad(valueVar, new IR.Shapes.Paddings(new(0, options.DynamicMaxTokens - numTokensVar), new(0, 0), new(0, 0)), PadMode.Constant, Tensor.Zeros(config.KVPrimType, [])) : (Expr)valueVar;
            var transValue = IR.F.Tensors.Transpose(padedValue, kvLayout.Select(x => (int)x).ToArray());
            var packedValue = kv_lanes.Length > 0 ? IR.F.Tensors.Pack(transValue, kv_lanes, kv_packed_axes) : transValue;
            updatedKVCache = IR.F.NN.UpdatePagedAttentionKVCache(
                packedValue,
                updatedKVCache,
                AttentionCacheKind.Value,
                layerId,
                kvLayout);

            // Apply attention for current layer
            packedQuery = IR.F.NN.PagedAttention(
                packedQuery,
                updatedKVCache,
                Tensor.Zeros(DataTypes.UInt8, [10 * 1024 * 1024]), // numTokens == 0 ? IR.F.Buffer.Uninitialized(config.KVPrimType, Nncase.TIR.MemoryLocation.Data, new RankedShape(numQHeads, numTokensVar, seqLens.Max() + 1)) : Tensor.Zeros(config.KVPrimType, [numQHeads, queryLens.Max(), seqLens.Max() + 1]), // [head_q, max_query_len, max_seq_len] + [head_q, max_query_len, 1]
                Tensor.FromScalar(1.0f).CastTo(config.KVPrimType, CastMode.KDefault),
                layerId,
                qLayout);
        }

        // unpack query.
        var unpacked = q_lanes.Length > 0 ? IR.F.Tensors.Unpack(packedQuery, q_lanes, q_packed_axes) : packedQuery;
        var untransed = IR.F.Tensors.Transpose(unpacked, qLayout.Select((x, i) => ((int)x, i)).OrderBy(p => p.Item1).Select(p => p.i).ToArray());
        var unpaded = options.DynamicShape ? IR.F.Tensors.Slice(untransed, new[] { 0 }, new Dimension[] { numTokensVar }, new[] { 0 }, new[] { 1 }) : untransed;
        Expr root = unpaded;

        if (options.TestUpdateKVCache)
        {
            root = IR.F.NN.GatherPagedAttentionKVCache(new[] { 0L }, updatedKVCache, numBlocks);
        }

        return (root, queryVar, kvVars, kvCacheObjVar);
    }

    public static (int[] Lanes, int[] Axes) GetQKVPackParams(IPagedAttentionConfig config, AttentionDimKind[] qLayout)
    {
        var lanes = new List<int>();
        var axes = new List<int>();
        for (int i = 0; i < config.PackedAxes.Count; i++)
        {
            if (config.PackedAxes[i] is PagedKVCacheDimKind.HeadDim or PagedKVCacheDimKind.NumKVHeads)
            {
                axes.Add(config.PackedAxes[i] switch
                {
                    PagedKVCacheDimKind.NumKVHeads => qLayout.IndexOf(AttentionDimKind.Head),
                    PagedKVCacheDimKind.HeadDim => qLayout.IndexOf(AttentionDimKind.Dim),
                    _ => throw new ArgumentOutOfRangeException(nameof(config)),
                });
                lanes.Add(config.Lanes[i]);
            }
        }

        return (lanes.ToArray(), axes.ToArray());
    }

    private Tensor GetSlotViewFromStorage(AttentionCacheKind kind, int layerId, int headId, Tensor slotId)
    {
        // slot_id : [len(topo) + 1].
        var slotIdValue = (long)slotId[slotId.Dimensions[^1] - 1];
        var blockIdValue = slotIdValue / PagedAttentionConfig.BlockSize;
        var blockOffset = slotIdValue % PagedAttentionConfig.BlockSize;

        var blockId = Tensor.Zeros<long>(slotId.Dimensions);
        slotId.AsContiguous().CopyTo(blockId);
        blockId[slotId.Dimensions[^1] - 1] = blockIdValue;

        var blockView = GetBlockViewFromStorage(kind, layerId, headId, blockId);
        var blockShape = blockView.Dimensions;
        var blockLayout = PagedAttentionConfig.BlockLayout;

        // PagedKVCacheDimKind[] defaultLayout = [PagedKVCacheDimKind.BlockSize, PagedKVCacheDimKind.HeadDim];
        int[] defaultLayout = [-1, -1, -1, 0, -1, 1];
        long[] defaultStarts = [blockOffset, 0];
        long[] defaultShape = [1, PagedAttentionConfig.HeadDim];
        if (PagedAttentionConfig.PackedAxes.IndexOf(PagedKVCacheDimKind.HeadDim) is int i && i != -1)
        {
            defaultShape[1] /= PagedAttentionConfig.Lanes[i];
        }

        var starts = blockLayout.Select(i => defaultStarts[defaultLayout[(int)i]]).ToArray();
        var shape = blockLayout.Select(i => defaultShape[defaultLayout[(int)i]]).ToArray();
        return blockView.View(starts, shape).Squeeze([blockLayout.IndexOf(PagedKVCacheDimKind.BlockSize)]);
    }

    private Tensor GetBlockViewFromStorage(AttentionCacheKind kind, int layerId, int headId, Tensor blockId)
    {
        // blockId: [len(topo) + 1].
        var blockIdValue = (long)blockId[blockId.Dimensions[^1] - 1];
        PagedKVCacheDimKind[] defaultLayout = [PagedKVCacheDimKind.NumBlocks, PagedKVCacheDimKind.NumLayers, PagedKVCacheDimKind.KV, PagedKVCacheDimKind.BlockSize, PagedKVCacheDimKind.NumKVHeads, PagedKVCacheDimKind.HeadDim];
        long[] defaultStarts = [blockIdValue, layerId, (long)kind, 0, (long)headId, 0];
        long[] defaultShape = [1, 1, 1, PagedAttentionConfig.BlockSize, 1, PagedAttentionConfig.HeadDim];
        if (PagedAttentionConfig.PackedAxes.IndexOf(PagedKVCacheDimKind.HeadDim) is int i && i != -1)
        {
            defaultShape[^1] /= PagedAttentionConfig.Lanes[i];
        }

        var starts = PagedAttentionConfig.CacheLayout.Select(i => defaultStarts[(int)i]).ToArray();
        var shape = PagedAttentionConfig.CacheLayout.Select(i => defaultShape[(int)i]).ToArray();
        var squeeze_axes = Enumerable.Range(0, PagedAttentionConfig.CacheLayout.Count).Where(i => PagedAttentionConfig.CacheLayout[i] is not (PagedKVCacheDimKind.BlockSize or PagedKVCacheDimKind.HeadDim)).Select(i => (long)i + PagedAttentionConfig.ShardingAxes.Count).ToArray();

        // process the topology.
        var topo_starts = Enumerable.Range(0, PagedAttentionConfig.ShardingAxes.Count).Select(i => (long)blockId[i]).ToArray();
        var topo_shape = Enumerable.Range(0, PagedAttentionConfig.ShardingAxes.Count).Select(i => 1L).ToArray();
        var topo_squeeze = Enumerable.Range(0, PagedAttentionConfig.ShardingAxes.Count).Select(i => (long)i).ToArray();

        var final_starts = topo_starts.Concat(starts).ToArray();
        var final_shape = topo_shape.Concat(shape).ToArray();
        var final_squeeze = topo_squeeze.Concat(squeeze_axes).ToArray();
        return KVCaches.View(final_starts, final_shape).Squeeze(final_squeeze);
    }

    public record class BuildKernelOptions(bool TestUpdateKVCache = false, bool DynamicShape = false, long DynamicMaxTokens = 128)
    {
    }
}

public sealed class PagedAttentionKVCacheJsonConverterFactory : JsonConverterFactory
{
    public override bool CanConvert(Type typeToConvert)
    {
        return typeof(IPagedAttentionKVCache).IsAssignableFrom(typeToConvert);
    }

    public override JsonConverter? CreateConverter(Type typeToConvert, JsonSerializerOptions options)
    {
        return new PagedAttentionKVCacheJsonConverter();
    }
}

internal sealed class PagedAttentionKVCacheJsonConverter : JsonConverter<IPagedAttentionKVCache>
{
    public override IPagedAttentionKVCache? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartObject)
        {
            throw new JsonException("Expected StartObject token");
        }

        using var doc = JsonDocument.ParseValue(ref reader);
        var root = doc.RootElement;

        var config = JsonSerializer.Deserialize<IPagedAttentionConfig>(
            root.GetProperty(nameof(RefPagedAttentionKVCache.Config)).GetRawText(),
            options)!;

        var numSeqs = root.GetProperty(nameof(RefPagedAttentionKVCache.NumSeqs)).GetInt32();
        var numTokens = root.GetProperty(nameof(RefPagedAttentionKVCache.NumTokens)).GetInt32();
        var numBlocks = root.GetProperty(nameof(RefPagedAttentionKVCache.NumBlocks)).GetInt32();

        var contextLens = JsonSerializer.Deserialize<Tensor<long>>(
            root.GetProperty(nameof(RefPagedAttentionKVCache.ContextLens)).GetRawText(),
            options)!;

        var seqLens = JsonSerializer.Deserialize<Tensor<long>>(
            root.GetProperty(nameof(RefPagedAttentionKVCache.SeqLens)).GetRawText(),
            options)!;

        var blockTable = JsonSerializer.Deserialize<Tensor<long>>(
            root.GetProperty(nameof(RefPagedAttentionKVCache.BlockTable)).GetRawText(),
            options)!;

        var slotMapping = JsonSerializer.Deserialize<Tensor<long>>(
            root.GetProperty(nameof(RefPagedAttentionKVCache.SlotMapping)).GetRawText(),
            options)!;

        var kvCaches = JsonSerializer.Deserialize<Tensor>(
            root.GetProperty(nameof(RefPagedAttentionKVCache.KVCaches)).GetRawText(),
            options)!;

        return new RefPagedAttentionKVCache(
            config,
            numSeqs,
            numTokens,
            contextLens,
            seqLens,
            blockTable,
            slotMapping,
            numBlocks,
            kvCaches);
    }

    public override void Write(Utf8JsonWriter writer, IPagedAttentionKVCache value, JsonSerializerOptions options)
    {
        var cache = (RefPagedAttentionKVCache)value;
        writer.WriteStartObject();

        writer.WritePropertyName(nameof(RefPagedAttentionKVCache.Config));
        JsonSerializer.Serialize(writer, (IPagedAttentionConfig)cache.Config, options);

        writer.WriteNumber(nameof(RefPagedAttentionKVCache.NumSeqs), cache.NumSeqs);
        writer.WriteNumber(nameof(RefPagedAttentionKVCache.NumTokens), cache.NumTokens);
        writer.WriteNumber(nameof(RefPagedAttentionKVCache.NumBlocks), cache.NumBlocks);

        writer.WritePropertyName(nameof(RefPagedAttentionKVCache.ContextLens));
        JsonSerializer.Serialize(writer, cache.ContextLens, options);

        writer.WritePropertyName(nameof(RefPagedAttentionKVCache.SeqLens));
        JsonSerializer.Serialize(writer, cache.SeqLens, options);

        writer.WritePropertyName(nameof(RefPagedAttentionKVCache.BlockTable));
        JsonSerializer.Serialize(writer, cache.BlockTable, options);

        writer.WritePropertyName(nameof(RefPagedAttentionKVCache.SlotMapping));
        JsonSerializer.Serialize(writer, cache.SlotMapping, options);

        writer.WritePropertyName(nameof(RefPagedAttentionKVCache.KVCaches));
        JsonSerializer.Serialize(writer, cache.KVCaches, options);

        writer.WriteEndObject();
    }
}
