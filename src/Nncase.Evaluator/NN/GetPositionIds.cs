// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using Tuple = System.Tuple;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="GetPositionIds"/>.
/// </summary>
public class GetPositionIdsEvaluator : IEvaluator<GetPositionIds>, ITypeInferencer<GetPositionIds>, ICostEvaluator<GetPositionIds>, IMetricEvaluator<GetPositionIds>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, GetPositionIds s)
    {
        var kvCache = context.GetArgumentValue(s, GetPositionIds.KVCache);
        var rangePair = GetRange(kvCache.AsTensor().Cast<Reference<IPagedAttentionKVCache>>());

        var positionIds = rangePair.Select(item => LinqUtility.Range(item.Item1, (int)(item.Item2 - item.Item1))).
            SelectMany(i => i).
            Select(i => (float)i).ToArray();
        return Value.FromTensor(Tensor.From(positionIds));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, GetPositionIds target)
    {
        var seqLen = (Dimension)context.GetArgument(target, GetPositionIds.SequenceLength);
        var tensorType = new TensorType(DataTypes.Float32, [seqLen]);
        return target.Placement.Rank == 0 ? tensorType : new DistributedType(tensorType, target.NdSBP, target.Placement);
    }

    public Cost Visit(ICostEvaluateContext context, GetPositionIds target)
    {
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(returnType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, GetPositionIds target)
    {
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
        };
    }

    private static List<Tuple<long, long>> GetRange(Tensor<Reference<IPagedAttentionKVCache>> kvCache)
    {
        var cache = kvCache.Single().Value;
        var range = new List<Tuple<long, long>>();
        for (int i = 0; i < cache.NumSeqs; i++)
        {
            var historyLen = cache.ContextLen(i);
            var seqLen = cache.SeqLen(i);
            range.Add(Tuple.Create(historyLen, seqLen)); // seqLen in silicaLLM is tokens num after this forward.
        }

        return range;
    }
}
