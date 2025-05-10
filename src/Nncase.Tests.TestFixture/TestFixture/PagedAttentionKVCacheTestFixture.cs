// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.Evaluator.OrtKIExtensions;

namespace Nncase.Tests.TestFixture;

public sealed class PagedAttentionKVCacheTestFixture
{
    public PagedAttentionKVCacheTestFixture(
        long[] queryLens,
        long[] seqLens,
        int numQHeads,
        int numKVHeads,
        int headDim,
        int blockSize,
        int numBlocks,
        Runtime.TypeCode kvPrimTypeCode,
        int numLayers,
        PagedKVCacheDimKind[] cacheLayout,
        PagedKVCacheDimKind[] packedAxes,
        PagedKVCacheDimKind[] shardingAxes,
        SBPSplit[] axisPolicies,
        AttentionDimKind[] qLayout,
        AttentionDimKind[] kLayout)
    {
        QueryLens = queryLens;
        SeqLens = seqLens;
        NumQHeads = numQHeads;
        ContextLens = queryLens.Zip(seqLens).Select(p => p.Second - p.First).ToArray();
        NumBlocks = numBlocks;
        QLayout = qLayout;
        KLayout = kLayout;
        var kvPrimType = DataType.FromTypeCode(kvPrimTypeCode);
        var lane = 128 / kvPrimType.SizeInBytes;
        Config = new PagedAttentionConfig(
            numLayers,
            numKVHeads,
            headDim,
            kvPrimType,
            blockSize,
            cacheLayout,
            packedAxes,
            new[] { lane },
            shardingAxes,
            axisPolicies);
    }

    public long[] QueryLens { get; }

    public long[] SeqLens { get; }

    public int NumQHeads { get; }

    public long[] ContextLens { get; }

    public int NumBlocks { get; }

    public AttentionDimKind[] QLayout { get; }

    public AttentionDimKind[] KLayout { get; }

    public PagedAttentionConfig Config { get; }

    /// <summary>
    /// query: [Hq,L,Ev], key: [Hk,L,Ev], value: [Hv,L,Ev].
    /// </summary>
    public static OrtKISharp.Tensor ScaledDotProductAttention(OrtKISharp.Tensor query, OrtKISharp.Tensor key, OrtKISharp.Tensor value, OrtKISharp.Tensor? attnMask = null, float dropoutP = 0.0f, bool isCausal = false, float? scale = null)
    {
        var curLen = query.Shape[^2];
        var histLen = key.Shape[^2];

        var scaleFactor = scale ?? 1 / MathF.Sqrt(query.Length);

        var attnBias = OrtKI.Expand(OrtKISharp.Tensor.FromScalar(0f), OrtKISharp.Tensor.MakeTensor([curLen, histLen]));

        if (isCausal)
        {
            var tempMask = OrtKISharp.Tensor.MakeTensor(Enumerable.Repeat(1.0f, (int)(curLen * histLen)).ToArray(), [curLen, histLen]);
            tempMask = OrtKI.Trilu(tempMask, OrtKISharp.Tensor.FromScalar<long>(0), 0);
            attnBias = OrtKI.Where(OrtKI.Equal(tempMask, OrtKISharp.Tensor.FromScalar(1.0f)), attnBias, OrtKI.Expand(OrtKISharp.Tensor.FromScalar(float.NegativeInfinity), OrtKISharp.Tensor.MakeTensor([curLen, histLen])));
        }

        if (attnMask != null)
        {
            throw new NotSupportedException("not support attnMask");
        }

        var perm = Enumerable.Range(0, key.Shape.Length).Select(i => (long)i).ToArray();
        (perm[^1], perm[^2]) = (perm[^2], perm[^1]);
        var attnWeight = OrtKI.MatMul(query, OrtKI.Transpose(key, perm)) * scaleFactor;
        attnWeight = attnWeight + attnBias;
        attnWeight = OrtKI.Softmax(attnWeight, -1);

        if (dropoutP > 0f)
        {
            throw new NotSupportedException("not support dropout");
        }

        return OrtKI.MatMul(attnWeight, value); // [Hq,L,Ev]
    }

    public static ReferenceResults PrepareReferenceResults(
        long[] queryLens,
        long[] seqLens,
        int numQHeads,
        int numKVHeads,
        int headDim,
        int numLayers,
        DataType primType)
    {
        var refQuerys = new List<OrtKISharp.Tensor>();
        var refOutputs = new List<OrtKISharp.Tensor>();
        var refKeys = new List<List<OrtKISharp.Tensor>>();
        var refValues = new List<List<OrtKISharp.Tensor>>();

        for (int req_id = 0; req_id < queryLens.Length; req_id++)
        {
            var seq_len = seqLens[req_id];
            var cur_len = queryLens[req_id];

            // Create query tensor
            var query = IR.F.Random.Normal(primType, new[] { numQHeads, cur_len, headDim }).Evaluate().AsTensor();
            var refQuery = query.ToOrtTensor();
            refQuerys.Add(refQuery);

            // Create key and value tensors for each layer
            var layerKeys = new List<OrtKISharp.Tensor>();
            var layerValues = new List<OrtKISharp.Tensor>();

            for (int layer = 0; layer < numLayers; layer++)
            {
                var key = IR.F.Random.Normal(primType, new[] { numKVHeads, seq_len, headDim }).Evaluate().AsTensor();
                var value = IR.F.Random.Normal(primType, new[] { numKVHeads, seq_len, headDim }).Evaluate().AsTensor();

                var refKey = key.ToOrtTensor();
                var refValue = value.ToOrtTensor();

                refQuery = ScaledDotProductAttention(
                    refQuery,
                    refKey, // use last layer's key
                    refValue, // use last layer's value
                    isCausal: true,
                    scale: 1.0f);

                layerKeys.Add(refKey);
                layerValues.Add(refValue);
            }

            refOutputs.Add(refQuery);
        }

        return new ReferenceResults(
            refQuerys,
            refOutputs,
            refKeys,
            refValues,
            numLayers,
            queryLens.Length);
    }

    public static TensorType GetQKVTensorType(ReadOnlySpan<long> defaultDims, IPagedAttentionConfig config)
    {
        var dims = defaultDims.ToArray();
        var lanes = new List<int>();
        for (int i = 0; i < config.PackedAxes.Count; i++)
        {
            if (config.PackedAxes[i] is PagedKVCacheDimKind.HeadDim or PagedKVCacheDimKind.NumKVHeads)
            {
                dims[config.PackedAxes[i] switch
                {
                    PagedKVCacheDimKind.NumKVHeads => 1,
                    PagedKVCacheDimKind.HeadDim => 2,
                    _ => throw new ArgumentOutOfRangeException(nameof(config)),
                }] /= config.Lanes[i];
                lanes.Add(config.Lanes[i]);
            }
        }

        return new TensorType(lanes.Count == 0 ? config.KVPrimType : new VectorType(config.KVPrimType, lanes.ToArray()), dims);
    }

    public static (int[] Lanes, int[] Axes) GetQKVPackParams(IPagedAttentionConfig config)
    {
        var lanes = new List<int>();
        var axes = new List<int>();
        for (int i = 0; i < config.PackedAxes.Count; i++)
        {
            if (config.PackedAxes[i] is PagedKVCacheDimKind.HeadDim or PagedKVCacheDimKind.NumKVHeads)
            {
                axes.Add(config.PackedAxes[i] switch
                {
                    PagedKVCacheDimKind.NumKVHeads => 1,
                    PagedKVCacheDimKind.HeadDim => 2,
                    _ => throw new ArgumentOutOfRangeException(nameof(config)),
                });
                lanes.Add(config.Lanes[i]);
            }
        }

        return (lanes.ToArray(), axes.ToArray());
    }

    /// <summary>
    /// all vars are [seq,head,dim] layout.
    /// </summary>
    public static TestKernel CreateTestKernel(long[] queryLens, int numQHeads, AttentionDimKind[] qLayout, AttentionDimKind[] kvLayout, IPagedAttentionConfig config)
    {
        var numTokens = queryLens.Sum();
        var defaultQDimenions = new long[] { numTokens, numQHeads, config.HeadDim };
        var queryVar = new Var("query", new TensorType(config.KVPrimType, defaultQDimenions));
        var kvVars = new List<Var[]>();
        var kvCacheObjVar = new Var("kvCache", TensorType.Scalar(
            new ReferenceType(new PagedAttentionKVCacheType { Config = config })));

        // Create vars for each layer
        var defaultKDimenions = new long[] { numTokens, config.NumKVHeads, config.HeadDim };
        for (int layerId = 0; layerId < config.NumLayers; layerId++)
        {
            var keyVar = new Var($"key_{layerId}", new TensorType(config.KVPrimType, defaultKDimenions));
            var valueVar = new Var($"value_{layerId}", new TensorType(config.KVPrimType, defaultKDimenions));
            kvVars.Add([keyVar, valueVar]);
        }

        // Build computation graph
        var (lanes, axes) = GetQKVPackParams(config);
        var packedQuery = lanes.Length > 0 ? IR.F.CPU.Pack(queryVar, lanes, axes) : queryVar;
        packedQuery = IR.F.Tensors.Transpose(packedQuery, qLayout.Select(x => (int)x).ToArray());
        for (int layerId = 0; layerId < config.NumLayers; layerId++)
        {
            var (keyVar, valueVar) = (kvVars[layerId][0], kvVars[layerId][1]);

            var packedKey = lanes.Length > 0 ? IR.F.CPU.Pack(keyVar, lanes, axes) : keyVar;
            packedKey = IR.F.Tensors.Transpose(packedKey, kvLayout.Select(x => (int)x).ToArray());
            var updatedKVCache = IR.F.NN.UpdatePagedAttentionKVCache(
                packedKey,
                kvCacheObjVar,
                AttentionCacheKind.Key,
                layerId);

            var packedValue = lanes.Length > 0 ? IR.F.CPU.Pack(valueVar, lanes, axes) : valueVar;
            packedValue = IR.F.Tensors.Transpose(packedValue, kvLayout.Select(x => (int)x).ToArray());
            updatedKVCache = IR.F.NN.UpdatePagedAttentionKVCache(
                packedValue,
                updatedKVCache,
                AttentionCacheKind.Value,
                layerId);

            // Apply attention for current layer
            packedQuery = IR.F.NN.PagedAttention(
                packedQuery,
                updatedKVCache,
                Const.FromTensor(Tensor.Zeros<byte>([1024])),
                layerId,
                qLayout);
        }

        // unpack query.
        var unpacked = lanes.Length > 0 ? IR.F.CPU.Unpack(packedQuery, lanes, axes) : packedQuery;
        var root = IR.F.Tensors.Transpose(unpacked, qLayout.Select((x, i) => ((int)x, i)).OrderBy(p => p.Item1).Select(p => p.i).ToArray());

        return new TestKernel(root, kvVars, kvCacheObjVar);
    }

    public static KVInputs PrepareKVInputs(long[] queryLens, long[] seqLens, long[] contextLens, int numBlocks, Placement placement, ReferenceResults referenceResults, IPagedAttentionConfig config)
    {
        // 0. create inputs tensor.
        var seqLensTensor = Tensor.From(seqLens);
        var contextLensTensor = Tensor.From(contextLens);

        // 1. create logical kv cache tensor.
        var logicalKVTensorType = config.GetLogicalTensorType(numBlocks, placement);
        var logicalKVCacheTensor = Tensor.Zeros(logicalKVTensorType.DType, logicalKVTensorType.Shape.ToValueArray());

        // 2. create temporary slotmapping for update hist key and value tensors.
        var blockTableTensorType = config.GetBlockTableTensorType(queryLens.Length, (int)seqLens.Max());
        var blockTableTensor = Tensor.Zeros(blockTableTensorType.DType, blockTableTensorType.Shape.ToValueArray()).Cast<long>();
        var alignedSeqLens = seqLens.Select(seqLen => MathUtility.AlignUp(seqLen, config.BlockSize)).ToArray();
        var alignedSeqStartLocs = alignedSeqLens.CumSum().ToArray();
        var alignedContextEndLocs = contextLens.Zip(alignedSeqStartLocs).Select(p => p.First + p.Second).ToArray();
        var kvInputs = new List<OrtKISharp.Tensor[]>();
        for (int layerId = 0; layerId < config.NumLayers; layerId++)
        {
            var layerKVInputs = new OrtKISharp.Tensor[2];
            kvInputs.Add(layerKVInputs);
            foreach (var kind in new[] { AttentionCacheKind.Key, AttentionCacheKind.Value })
            {
                var histOrtTensorList = new List<OrtKISharp.Tensor>();
                var curOrtTensorList = new List<OrtKISharp.Tensor>();
                for (int seqId = 0; seqId < queryLens.Length; seqId++)
                {
                    var contextLen = contextLens[seqId];
                    var queryLen = queryLens[seqId];
                    var refKVTensor = referenceResults.GetKeyValue(seqId, layerId, kind);
                    var refKVTensors = OrtKI.Split(refKVTensor, new long[] { contextLen, queryLen }, 1);
                    var (histOrtTensor, curOrtTensor) = (refKVTensors[0], refKVTensors[1]);
                    histOrtTensorList.Add(histOrtTensor);
                    curOrtTensorList.Add(curOrtTensor);
                }

                var histOrtTensors = OrtKI.Concat(histOrtTensorList.ToArray(), 1); // [heads, seq_len, head_dim]
                var curOrtTensors = OrtKI.Concat(curOrtTensorList.ToArray(), 1);
                layerKVInputs[(int)kind] = curOrtTensors;

                // pack tensors.
                for (int i = 0; i < config.PackedAxes.Count; i++)
                {
                    if (config.PackedAxes[i] is PagedKVCacheDimKind.HeadDim or PagedKVCacheDimKind.NumKVHeads)
                    {
                        histOrtTensors = histOrtTensors.Pack(
                            config.Lanes[i],
                            config.PackedAxes[i] switch { PagedKVCacheDimKind.NumKVHeads => 1, PagedKVCacheDimKind.HeadDim => 2, _ => throw new ArgumentOutOfRangeException(nameof(config)) });
                    }
                }

                var histTensor = Tensor.FromBytes(new TensorType(config.KVType, histOrtTensors.Shape.SkipLast(config.PackedAxes.Count).ToArray()), histOrtTensors.BytesBuffer.ToArray());

                // assign slot id.
                if (contextLens.Sum() != (int)histOrtTensors.Shape[1])
                {
                    throw new ArgumentOutOfRangeException(nameof(contextLens));
                }

                var tempSlotMappingTensorType = config.GetSlotMappingTensorType((int)contextLens.Sum());
                var tempSlotMappingTensor = Tensor.Zeros(tempSlotMappingTensorType.DType, tempSlotMappingTensorType.Shape.ToValueArray()).Cast<long>();

                for (long tokenId = 0, seqId = 0; seqId < queryLens.Length; seqId++)
                {
                    for (long logical_slot_id = alignedSeqStartLocs[seqId]; logical_slot_id < alignedSeqStartLocs[seqId] + alignedContextEndLocs[seqId]; logical_slot_id++)
                    {
                        var indices = new long[] { tokenId, 0 };
                        var physical_slot_id = logical_slot_id;

                        // process sharding axes.
                        for (int shard_id = 0; shard_id < config.ShardingAxes.Count; shard_id++)
                        {
                            int value;
                            switch (config.ShardingAxes[shard_id])
                            {
                                case PagedKVCacheDimKind.NumBlocks:
                                    value = (int)Math.DivRem(physical_slot_id, config.AxisPolicies[shard_id].Axes.Select(axis => placement.Hierarchy[axis]).Product(), out physical_slot_id);
                                    break;
                                case PagedKVCacheDimKind.HeadDim:
                                    value = -1;
                                    break;
                                default:
                                    throw new ArgumentOutOfRangeException(nameof(config));
                            }

                            tempSlotMappingTensor[indices] = value;
                            indices[^1]++;
                        }

                        tempSlotMappingTensor[indices] = physical_slot_id;
                    }
                }

                // start updating
                var tmpKVCacheObj = new RefPagedAttentionKVCache(
                    config,
                    queryLens.Length,
                    (int)queryLens.Sum(),
                    contextLensTensor,
                    seqLensTensor,
                    blockTableTensor,
                    tempSlotMappingTensor,
                    numBlocks,
                    logicalKVCacheTensor);

                for (int tokenId = 0; tokenId < contextLens.Sum(); tokenId++)
                {
                    var slotId = tmpKVCacheObj.GetSlotId(tokenId);
                    for (int headId = 0; headId < config.NumKVHeads; headId++)
                    {
                        var slot = histTensor.View([headId, tokenId, 0], [1, 1, histTensor.Dimensions[^1]]); // [heads, seq_len, head_dim]
                        tmpKVCacheObj.UpdateSlot(kind, layerId, headId, slotId, slot);
                    }
                }
            }
        }

        // 3. create block table/slot mapping tensor.
        for (int seqId = 0; seqId < seqLens.Length; seqId++)
        {
            var num_blocks = MathUtility.CeilDiv(seqLens[seqId], config.BlockSize);
            for (long j = 0, logicalSlotId = alignedSeqStartLocs[seqId]; logicalSlotId < alignedSeqStartLocs[seqId] + alignedSeqLens[seqId]; logicalSlotId += config.BlockSize)
            {
                var logicalBlockId = logicalSlotId / config.BlockSize;
                var indices = new long[] { seqId, j, 0 };
                long physicalBlockId = logicalBlockId;

                for (int topoId = 0; topoId < config.ShardingAxes.Count; topoId++)
                {
                    switch (config.ShardingAxes[topoId])
                    {
                        case PagedKVCacheDimKind.NumBlocks:
                            var value = (int)Math.DivRem(physicalBlockId, config.AxisPolicies[topoId].Axes.Select(axis => placement.Hierarchy[axis]).Product(), out physicalBlockId);
                            blockTableTensor[indices] = value;
                            break;
                        case PagedKVCacheDimKind.HeadDim:
                            blockTableTensor[indices] = -1;
                            break;
                        default:
                            throw new ArgumentOutOfRangeException(nameof(config));
                    }

                    indices[^1]++;
                }

                blockTableTensor[indices] = j;
            }
        }

        var slotMappingTensorType = config.GetSlotMappingTensorType((int)queryLens.Sum());
        var slotMappingTensor = Tensor.Zeros(slotMappingTensorType.DType, slotMappingTensorType.Shape.ToValueArray()).Cast<long>();

        for (long tokenId = 0, seqId = 0; seqId < seqLens.Length; seqId++)
        {
            for (long logicalSlotId = alignedContextEndLocs[seqId]; logicalSlotId < alignedContextEndLocs[seqId] + queryLens[seqId]; logicalSlotId++)
            {
                var indices = new long[] { tokenId, 0 };
                var physicalSlotId = logicalSlotId;

                for (int topoId = 0; topoId < config.ShardingAxes.Count; topoId++)
                {
                    switch (config.ShardingAxes[topoId])
                    {
                        case PagedKVCacheDimKind.NumBlocks:
                            var value = (int)Math.DivRem(physicalSlotId, config.AxisPolicies[topoId].Axes.Select(axis => placement.Hierarchy[axis]).Product(), out physicalSlotId);
                            slotMappingTensor[indices] = value;
                            break;
                        case PagedKVCacheDimKind.HeadDim:
                            slotMappingTensor[indices] = -1;
                            break;
                        default:
                            throw new ArgumentOutOfRangeException(nameof(config));
                    }

                    indices[^1]++;
                }

                slotMappingTensor[indices] = physicalSlotId;
                tokenId++;
            }
        }

        // final inputs
        var refQuerys = new List<OrtKISharp.Tensor>();
        var kvcacheObj = new RefPagedAttentionKVCache(
            config,
            queryLens.Length,
            (int)queryLens.Sum(),
            contextLensTensor,
            seqLensTensor,
            blockTableTensor,
            slotMappingTensor,
            numBlocks,
            logicalKVCacheTensor);
        return new(kvInputs, kvcacheObj);
    }

    public sealed record TestKernel(Expr Root, List<Var[]> KVVars, Var KVCacheObjVar);

    public sealed record KVInputs(List<OrtKISharp.Tensor[]> KVTensors, RefPagedAttentionKVCache KVCacheObj);

    public sealed class ReferenceResults
    {
        private readonly List<OrtKISharp.Tensor> _refQuerys;
        private readonly List<OrtKISharp.Tensor> _refOutputs;
        private readonly List<List<OrtKISharp.Tensor>> _refKeys;
        private readonly List<List<OrtKISharp.Tensor>> _refValues;
        private readonly int _numLayers;
        private readonly int _numSeqs;

        public ReferenceResults(
            List<OrtKISharp.Tensor> refQuerys,
            List<OrtKISharp.Tensor> refOutputs,
            List<List<OrtKISharp.Tensor>> refKeys,
            List<List<OrtKISharp.Tensor>> refValues,
            int numLayers,
            int numSeqs)
        {
            _refQuerys = refQuerys;
            _refOutputs = refOutputs;
            _refKeys = refKeys;
            _refValues = refValues;
            _numLayers = numLayers;
            _numSeqs = numSeqs;
        }

        public OrtKISharp.Tensor GetQuery(int seqId)
        {
            if (seqId >= _numSeqs)
            {
                throw new ArgumentOutOfRangeException(nameof(seqId));
            }

            return _refQuerys[seqId];
        }

        public OrtKISharp.Tensor GetOutput(int seqId)
        {
            if (seqId >= _numSeqs)
            {
                throw new ArgumentOutOfRangeException(nameof(seqId));
            }

            return _refOutputs[seqId];
        }

        /// <summary>
        /// [heads, seq_len, head_dim].
        /// </summary>
        public OrtKISharp.Tensor GetKeyValue(int seqId, int layerId, AttentionCacheKind kind)
        {
            if (seqId >= _numSeqs)
            {
                throw new ArgumentOutOfRangeException(nameof(seqId));
            }

            if (layerId >= _numLayers)
            {
                throw new ArgumentOutOfRangeException(nameof(layerId));
            }

            return kind == AttentionCacheKind.Key ? _refKeys[seqId][layerId] : _refValues[seqId][layerId];
        }
    }
}
