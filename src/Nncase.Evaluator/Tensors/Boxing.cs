// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
#pragma warning disable SA1010, SA1008
using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Evaluator.Tensors;

public sealed class BoxingEvaluator : ITypeInferencer<Boxing>, ICostEvaluator<Boxing>
{
    private const int _burstLength = 256;

    public IRType Visit(ITypeInferenceContext context, Boxing target)
    {
        return target.NewType;
    }

    public Cost Visit(ICostEvaluateContext context, Boxing target)
    {
        var inType = context.GetArgumentType<IRType>(target, Boxing.Input);
        var returnType = context.GetReturnType<IRType>();
        Cost cost;
        switch (inType, returnType)
        {
            case (TensorType tensorType, DistributedType distTensorType):
                cost = new Cost()
                {
                    [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(tensorType),
                    [CostFactorNames.MemoryStore] = (UInt128)((float)CostUtility.GetMemoryAccess(distTensorType) / DistributedUtilities.GetDividedTensorEfficiency(distTensorType, _burstLength)),
                };
                break;
            case (DistributedType distTensorType, TensorType tensorType):
                cost = new Cost()
                {
                    [CostFactorNames.MemoryLoad] = (UInt128)((float)CostUtility.GetMemoryAccess(distTensorType) / DistributedUtilities.GetDividedTensorEfficiency(distTensorType, _burstLength)),
                    [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(tensorType),
                };
                break;
            case (DistributedType { Placement: { Rank: 1 } } din, DistributedType { Placement: { Rank: 2 } } dout):
                {
                    // shared tensor broadcast to local.
                    var a = DistributedUtilities.GetDividedTensorType(din);
                    var b = DistributedUtilities.GetDividedTensorType(dout);
                    cost = new Cost()
                    {
                        [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(a),
                        [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(b),
                    };
                }

                break;
            case (DistributedType { NdSbp: [(SBPSplit or SBPBroadCast), SBPPartialSum], Placement.Rank: 2 } din,
                  DistributedType { NdSbp: [(SBPSplit or SBPBroadCast), SBPBroadCast], Placement.Rank: 2 }):
                {
                    // reduce scater and broadcast (ring reduce)
                    _ = DistributedUtilities.GetDividedTensorType(din);
                    var v = (int)CostUtility.GetMemoryAccess(din.TensorType);
                    var comm = (din.Placement.Hierarchy[1] - 1) * (v / din.Placement.Hierarchy[1]) * 2; // reduce-scatter + gather
                    cost = new Cost()
                    {
                        [CostFactorNames.MemoryLoad] = (UInt128)comm,
                        [CostFactorNames.MemoryStore] = (UInt128)comm,
                    };
                }

                break;
            case (DistributedType a, DistributedType b) when a.Placement == b.Placement && a.NdSbp != b.NdSbp:
                {
                    cost = new Cost()
                    {
                        [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(a) + CostUtility.GetMemoryAccess(b),
                        [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(a.TensorType),
                    };
                }

                break;
            case (DistributedType a, DistributedType b) when a == b:
                throw new InvalidOperationException($"the boxing inType == outType");
            default:
                throw new NotSupportedException($"{inType} {returnType}");
        }

        return cost;
    }
}
