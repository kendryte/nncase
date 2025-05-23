// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.Evaluator.NN;

public sealed class RefPagedAttentionScheduler
{
    private readonly IPagedAttentionConfig _config;
    private readonly int _numBlocks;
    private readonly int _maxModelLen;
    private readonly Dictionary<long, SessionInfo> _sessionInfos = new Dictionary<long, SessionInfo>();
    private readonly Tensor _kvCaches;
    private readonly Placement _placement;

    /// <summary>
    /// Initializes a new instance of the <see cref="RefPagedAttentionScheduler"/> class.
    /// </summary>
    /// <param name="config">Paged attention configuration.</param>
    /// <param name="numBlocks">Number of blocks.</param>
    /// <param name="maxModelLen">Maximum model length.</param>
    /// <param name="hierarchy">Hardware Hierarchy information.</param>
    public RefPagedAttentionScheduler(IPagedAttentionConfig config, int numBlocks, int maxModelLen, int[] hierarchy)
    {
        _config = config;
        _numBlocks = numBlocks;
        _maxModelLen = maxModelLen;

        if (_maxModelLen % _config.BlockSize != 0)
        {
            throw new ArgumentException("Max model length must be a multiple of block size.");
        }

        _placement = new Placement(hierarchy, string.Empty);
        var tensorType = _config.GetLogicalShardTensorType(numBlocks, _placement);
        _kvCaches = Tensor.Zeros(tensorType.DType, tensorType.Shape.ToValueArray());
    }

    /// <summary>
    /// Schedule paged attention.
    /// </summary>
    /// <param name="sessionIds">Session IDs tensor.</param>
    /// <param name="queryLens">Tokens count tensor.</param>
    /// <returns>Paged attention KV cache.</returns>
    public RefPagedAttentionKVCache Schedule(long[] sessionIds, long[] queryLens)
    {
        if (sessionIds.Length != queryLens.Length)
        {
            throw new ArgumentException("Session IDs and query lengths must have the same length.");
        }

        int numSeqs = sessionIds.Length;
        int numTokens = queryLens.Sum(x => (int)x);

        var seqLens = new long[numSeqs];
        var contextLens = new long[numSeqs];
        var seqLogicalSlotIds = new long[numSeqs][];
        long maxSeqLen = 0;

        // Process each session
        for (int seqId = 0; seqId < numSeqs; seqId++)
        {
            long sessionId = sessionIds[seqId];
            long queryLen = queryLens[seqId];

            if (!_sessionInfos.TryGetValue(sessionId, out var info))
            {
                info = new SessionInfo
                {
                    SlotStart = sessionId * _maxModelLen,
                    SlotEnd = (sessionId + 1) * _maxModelLen,
                    ContextLen = 0,
                };

                _sessionInfos.Add(sessionId, info);
            }

            if (info.SlotEnd > _numBlocks * _config.BlockSize)
            {
                throw new InvalidOperationException("Can't allocate KV cache for new session!");
            }

            contextLens[seqId] = info.ContextLen;
            info.ContextLen += queryLen;
            seqLens[seqId] = contextLens[seqId] + queryLen;

            if (seqLens[seqId] > _maxModelLen)
            {
                throw new InvalidOperationException("The sequence length is larger than max model length!");
            }

            maxSeqLen = System.Math.Max(maxSeqLen, seqLens[seqId]);

            seqLogicalSlotIds[seqId] = new long[queryLen];
            for (int j = 0; j < queryLen; j++)
            {
                seqLogicalSlotIds[seqId][j] = info.SlotStart + contextLens[seqId] + j;
            }
        }

        // start create tensors.
        // int NumSeqs
        // int NumTokens,
        var seqLensTensor = new Tensor<long>(seqLens, [numSeqs]);
        var contextLensTensor = new Tensor<long>(contextLens, [numSeqs]);
        var blockTableTensorType = _config.GetBlockTableTensorType(numSeqs, (int)maxSeqLen);
        var blockTableTensor = Tensor.Zeros(blockTableTensorType.DType, blockTableTensorType.Shape.ToValueArray()).Cast<long>();
        for (int seqId = 0; seqId < numSeqs; seqId++)
        {
            var logicalSlotIds = seqLogicalSlotIds[seqId];
            var info = _sessionInfos[sessionIds[seqId]];
            for (long itemId = 0, logicalSlotId = info.SlotStart; logicalSlotId < Utilities.MathUtility.AlignUp(info.SlotStart + seqLens[seqId], _config.BlockSize); logicalSlotId += _config.BlockSize, itemId++)
            {
                var logicalBlockId = logicalSlotId / _config.BlockSize;
                RefPagedAttentionKVCache.MaterializeBlockTable(blockTableTensor, [seqId, itemId, 0], logicalBlockId, _numBlocks, _placement, _config);
            }
        }

        var slotMappingTensorType = _config.GetSlotMappingTensorType(numTokens);
        var slotMappingTensor = Tensor.Zeros(slotMappingTensorType.DType, slotMappingTensorType.Shape.ToValueArray()).Cast<long>();
        for (long tokenId = 0, seqId = 0; seqId < seqLens.Length; seqId++)
        {
            var info = _sessionInfos[sessionIds[seqId]];
            var contextLen = contextLens[seqId];
            for (long logicalSlotId = info.SlotStart + contextLen; logicalSlotId < info.SlotStart + contextLen + queryLens[seqId]; logicalSlotId++)
            {
                RefPagedAttentionKVCache.MaterializeSlotMappingId(slotMappingTensor, [tokenId, 0], logicalSlotId, _numBlocks, _placement, _config);
                tokenId++;
            }
        }

        return new RefPagedAttentionKVCache(_config, numSeqs, numTokens, contextLensTensor, seqLensTensor, blockTableTensor, slotMappingTensor, _numBlocks, _kvCaches);
    }

    public IR.Function CreateTestFunction(int numQHeads, AttentionDimKind[] qLayout, AttentionDimKind[] kvLayout)
    {
        var (root, queryVar, kVVars, kVCacheObjVar) = RefPagedAttentionKVCache.BuildPagedAttentionKernel([], [_maxModelLen], numQHeads, _numBlocks, qLayout, kvLayout, _config);
        return new IR.Function(root, new Var[] { queryVar }.Concat(kVVars.SelectMany(i => i).ToArray()).Concat(new Var[] { kVCacheObjVar }).ToArray());
    }

    private class SessionInfo
    {
        public long SlotStart { get; set; }

        public long SlotEnd { get; set; }

        public long ContextLen { get; set; }
    }
}
