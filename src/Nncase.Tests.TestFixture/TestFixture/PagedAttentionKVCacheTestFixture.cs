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

        OrtKISharp.Tensor scaleFactor = scale ?? 1 / MathF.Sqrt(query.Length);
        scaleFactor = scaleFactor.Cast(query.DataType);

        var attnBias = OrtKI.Expand(OrtKISharp.Tensor.FromScalar(0f), OrtKISharp.Tensor.MakeTensor([curLen, histLen]));

        if (isCausal)
        {
            var tempMask = OrtKISharp.Tensor.MakeTensor(Enumerable.Repeat(1.0f, (int)(curLen * histLen)).ToArray(), [curLen, histLen]);
            tempMask = OrtKI.Trilu(tempMask, OrtKISharp.Tensor.FromScalar<long>(0), 0);
            attnBias = OrtKI.Where(OrtKI.Equal(tempMask, OrtKISharp.Tensor.FromScalar(1.0f)), attnBias, OrtKI.Expand(OrtKISharp.Tensor.FromScalar(float.NegativeInfinity), OrtKISharp.Tensor.MakeTensor([curLen, histLen])));
        }

        attnBias = attnBias.Cast(query.DataType);

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
        DataType primType,
        bool randomData = true)
    {
        var refQuerys = new List<OrtKISharp.Tensor>();
        var refOutputs = new List<OrtKISharp.Tensor>();
        var refKeys = new List<List<OrtKISharp.Tensor>>();
        var refValues = new List<List<OrtKISharp.Tensor>>();
        var refDataValue = 1f;

        for (int req_id = 0; req_id < queryLens.Length; req_id++)
        {
            var seq_len = seqLens[req_id];
            var cur_len = queryLens[req_id];

            // Create query tensor
            var qDimension = new[] { numQHeads, cur_len, headDim };
            Tensor query = randomData ? IR.F.Random.Normal(primType, qDimension).Evaluate().AsTensor() : ReferenceInputGenerator(qDimension, ref refDataValue).CastTo(primType);

            var refQuery = query.ToOrtTensor();
            refQuerys.Add(refQuery);

            // Create key and value tensors for each layer
            var layerKeys = new List<OrtKISharp.Tensor>();
            var layerValues = new List<OrtKISharp.Tensor>();
            refKeys.Add(layerKeys);
            refValues.Add(layerValues);

            for (int layer = 0; layer < numLayers; layer++)
            {
                var kvDimension = new[] { numKVHeads, seq_len, headDim };
                var key = randomData ? IR.F.Random.Normal(primType, kvDimension).Evaluate().AsTensor() : ReferenceInputGenerator(kvDimension, ref refDataValue).CastTo(primType);
                var value = randomData ? IR.F.Random.Normal(primType, kvDimension).Evaluate().AsTensor() : ReferenceInputGenerator(kvDimension, ref refDataValue).CastTo(primType);

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

    /// <summary>
    /// all vars are [seq,head,dim] layout.
    /// </summary>
    public static TestKernel CreateTestKernel(long[] queryLens, int numQHeads, int numBlocks, AttentionDimKind[] qLayout, AttentionDimKind[] kvLayout, IPagedAttentionConfig config, bool testUpdateKVCache = false)
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
        var (q_lanes, q_packed_axes) = GetQKVPackParams(config, qLayout);
        var (kv_lanes, kv_packed_axes) = GetQKVPackParams(config, kvLayout);
        var transedQuery = IR.F.Tensors.Transpose(queryVar, qLayout.Select(x => (int)x).ToArray());
        var packedQuery = q_lanes.Length > 0 ? IR.F.CPU.Pack(transedQuery, q_lanes, q_packed_axes) : transedQuery;
        Expr updatedKVCache = None.Default;
        for (int layerId = 0; layerId < config.NumLayers; layerId++)
        {
            var (keyVar, valueVar) = (kvVars[layerId][0], kvVars[layerId][1]);

            var transedKey = IR.F.Tensors.Transpose(keyVar, kvLayout.Select(x => (int)x).ToArray());
            var packedKey = kv_lanes.Length > 0 ? IR.F.CPU.Pack(transedKey, kv_lanes, kv_packed_axes) : transedKey;
            updatedKVCache = IR.F.NN.UpdatePagedAttentionKVCache(
                packedKey,
                kvCacheObjVar,
                AttentionCacheKind.Key,
                layerId,
                kvLayout);

            var transValue = IR.F.Tensors.Transpose(valueVar, kvLayout.Select(x => (int)x).ToArray());
            var packedValue = kv_lanes.Length > 0 ? IR.F.CPU.Pack(transValue, kv_lanes, kv_packed_axes) : transValue;
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
                Const.FromTensor(Tensor.Zeros<byte>([1024])),
                layerId,
                qLayout);
        }

        // unpack query.
        var unpacked = q_lanes.Length > 0 ? IR.F.CPU.Unpack(packedQuery, q_lanes, q_packed_axes) : packedQuery;
        Expr root = IR.F.Tensors.Transpose(unpacked, qLayout.Select((x, i) => ((int)x, i)).OrderBy(p => p.Item1).Select(p => p.i).ToArray());

        if (testUpdateKVCache)
        {
            root = IR.F.NN.GatherPagedAttentionKVCache(new[] { 0L }, updatedKVCache, numBlocks);
        }

        return new TestKernel(root, queryVar, kvVars, kvCacheObjVar);
    }

    public static KVInputs PrepareKVInputs(long[] queryLens, long[] seqLens, long[] contextLens, int numBlocks, Placement placement, ReferenceResults referenceResults, IPagedAttentionConfig config)
    {
        // 0. create inputs tensor.
        var seqLensTensor = Tensor.From(seqLens);
        var contextLensTensor = Tensor.From(contextLens);

        // 1. create logical kv cache tensor.
        var logicalKVTensorType = config.GetLogicalShardTensorType(numBlocks, placement);
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
                var contextTokens = contextLens.Sum();
                if (contextTokens != (int)histOrtTensors.Shape[1])
                {
                    throw new ArgumentOutOfRangeException(nameof(contextLens));
                }

                if (contextTokens == 0)
                {
                    continue;
                }

                var tempSlotMappingTensorType = config.GetSlotMappingTensorType((int)contextLens.Sum());
                var tempSlotMappingTensor = Tensor.Zeros(tempSlotMappingTensorType.DType, tempSlotMappingTensorType.Shape.ToValueArray()).Cast<long>();

                for (long tokenId = 0, seqId = 0; seqId < queryLens.Length; seqId++)
                {
                    for (long logical_slot_id = alignedSeqStartLocs[seqId]; logical_slot_id < alignedSeqStartLocs[seqId] + contextLens[seqId]; logical_slot_id++)
                    {
                        var indices = new long[] { tokenId, 0 };
                        PrepareSlotMappingId(tempSlotMappingTensor, [tokenId, 0], logical_slot_id, numBlocks, placement, config);
                        tokenId++;
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
                        var slot = histTensor.View([headId, tokenId, 0], [1, 1, histTensor.Dimensions[^1]]).Squeeze(0, 1); // [heads, seq_len, head_dim]
                        tmpKVCacheObj.UpdateSlot(kind, layerId, headId, slotId, slot);
                    }
                }
            }
        }

        // 3. create block table/slot mapping tensor.
        for (int seqId = 0; seqId < seqLens.Length; seqId++)
        {
            for (long j = 0, logicalSlotId = alignedSeqStartLocs[seqId]; logicalSlotId < alignedSeqStartLocs[seqId] + alignedSeqLens[seqId]; logicalSlotId += config.BlockSize, j++)
            {
                var logicalBlockId = logicalSlotId / config.BlockSize;
                var indices = new long[] { seqId, j, 0 };
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
                            var value = (int)Math.DivRem(physicalBlockId, numBlockTile, out physicalBlockId);
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
        }

        var slotMappingTensorType = config.GetSlotMappingTensorType((int)queryLens.Sum());
        var slotMappingTensor = Tensor.Zeros(slotMappingTensorType.DType, slotMappingTensorType.Shape.ToValueArray()).Cast<long>();

        for (long tokenId = 0, seqId = 0; seqId < seqLens.Length; seqId++)
        {
            for (long logicalSlotId = alignedContextEndLocs[seqId]; logicalSlotId < alignedContextEndLocs[seqId] + queryLens[seqId]; logicalSlotId++)
            {
                PrepareSlotMappingId(slotMappingTensor, [tokenId, 0], logicalSlotId, numBlocks, placement, config);
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

    public static void PrepareSlotMappingId(Tensor<long> slotMappingTensor, long[] indices, long logicalSlotId, int numBlocks, Placement placement, IPagedAttentionConfig config)
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
                    var value = (int)Math.DivRem(physicalSlotId, numBlockTile, out physicalSlotId);
                    slotMappingTensor[indices] = value;
                    break;
                case PagedKVCacheDimKind.NumKVHeads when config.AxisPolicies[shardId].Axes.Count == 1:
                    slotMappingTensor[indices] = -1; // todo should matching the kv sharding.
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(config));
            }

            indices[^1]++;
        }

        slotMappingTensor[indices] = physicalSlotId;
    }

    public static Tensor<float> ReferenceInputGenerator(long[] dimensions, ref float startValue)
    {
        // head,seq,dim.
        var buffer = new float[dimensions.Product()];
        var strides = TensorUtilities.GetStrides(dimensions);
        var span = buffer.AsSpan();
        for (int headId = 0; headId < dimensions[0]; headId++)
        {
            for (int tokenId = 0; tokenId < dimensions[1]; tokenId++)
            {
                var subSpan = span.Slice((int)((headId * strides[0]) + (tokenId * strides[1])), (int)strides[1]);
                subSpan.Fill(startValue++);
            }
        }

        return Tensor.From(buffer, dimensions);
    }

    public sealed record TestKernel(Expr Root, Var QueryVar, List<Var[]> KVVars, Var KVCacheObjVar);

    public sealed record KVInputs(List<OrtKISharp.Tensor[]> KVTensors, RefPagedAttentionKVCache KVCacheObj)
    {
        public Tensor GetKeyValueTensor(int layerId, int kind)
        {
            return OrtKI.Transpose(KVTensors[layerId][kind], [1, 0, 2]).ToTensor();
        }
    }

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

        public Tensor GetQueryTensor()
        {
            return OrtKI.Transpose(OrtKI.Concat(_refQuerys.ToArray(), 1), [1, 0, 2]).ToTensor();
        }

        public Tensor GetOutputTensor()
        {
            return OrtKI.Transpose(OrtKI.Concat(_refOutputs.ToArray(), 1), [1, 0, 2]).ToTensor();
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
