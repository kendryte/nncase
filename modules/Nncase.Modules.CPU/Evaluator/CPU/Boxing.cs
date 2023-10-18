// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
#pragma warning disable SA1010, SA1008
using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.Utilities;

namespace Nncase.Evaluator.CPU;

public sealed class BoxingEvaluator : ITypeInferencer<Boxing>, ICostEvaluator<Boxing>, IEvaluator<Boxing>
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
        var cost = new Cost() { [CostFactorNames.MemoryLoad] = 0, [CostFactorNames.MemoryStore] = 0 };
        switch (inType, returnType)
        {
            case (TensorType tensorType, DistributedType distTensorType):
                cost = new Cost()
                {
                    [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(tensorType),
                    [CostFactorNames.MemoryStore] = (UInt128)((float)CostUtility.GetMemoryAccess(distTensorType) / DistributedUtility.GetDividedTensorEfficiency(distTensorType, _burstLength)),
                };
                break;
            case (DistributedType distTensorType, TensorType tensorType):
                cost = new Cost()
                {
                    [CostFactorNames.MemoryLoad] = (UInt128)((float)CostUtility.GetMemoryAccess(distTensorType) / DistributedUtility.GetDividedTensorEfficiency(distTensorType, _burstLength)),
                    [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(tensorType),
                };
                break;

            case (DistributedType a, DistributedType b) when a.Placement == b.Placement && a.NdSBP != b.NdSBP:
                {
                    var fullLoadStore = new Cost()
                    {
                        [CostFactorNames.MemoryStore] = (UInt128)((float)CostUtility.GetMemoryAccess(a) / DistributedUtility.GetDividedTensorEfficiency(a, _burstLength)),
                        [CostFactorNames.MemoryLoad] = (UInt128)((float)CostUtility.GetMemoryAccess(b) / DistributedUtility.GetDividedTensorEfficiency(b, _burstLength)),
                    };

                    float scatterPart = 1;
                    float gatherPart = 1;
                    for (int i = 0; i < a.Placement.Rank; i++)
                    {
                        switch (a.NdSBP[i], b.NdSBP[i])
                        {
                            case (SBPSplit { Axis: int ax }, SBP sbpout):
                                switch (sbpout)
                                {
                                    case SBPSplit { Axis: int bx }:
                                        if (ax != bx)
                                        {
                                            // when split different axis, need global load store.
                                            return fullLoadStore;
                                        }

                                        break;
                                    case SBPBroadCast:
                                        scatterPart *= a.Placement.Hierarchy[i];
                                        gatherPart *= a.Placement.Hierarchy[i];
                                        break;
                                    default:
                                        throw new NotSupportedException("split to partial");
                                }

                                break;
                            case (SBPBroadCast, SBPBroadCast or SBPSplit):
                                // no cost.
                                break;
                            case (SBPPartialSum, SBP sbpout):
                                switch (sbpout)
                                {
                                    case SBPPartialSum:
                                        break;
                                    case SBPBroadCast or SBPSplit:
                                        gatherPart *= a.Placement.Hierarchy[i];
                                        if (i == 0)
                                        {
                                            scatterPart *= a.Placement.Hierarchy[i];
                                        }

                                        break;
                                }

                                break;
                            default:
                                throw new NotSupportedException($"{a} to {b}");
                        }
                    }

                    if (gatherPart > 1f)
                    {
                        cost += new Cost()
                        {
                            [CostFactorNames.MemoryStore] = (UInt128)((gatherPart - 1) * (float)CostUtility.GetMemoryAccess(DistributedUtility.GetDividedTensorType(a)) / gatherPart),
                        };
                    }

                    if (scatterPart > 1f)
                    {
                        cost += new Cost()
                        {
                            [CostFactorNames.MemoryLoad] = (UInt128)((scatterPart - 1) * (float)CostUtility.GetMemoryAccess(DistributedUtility.GetDividedTensorType(b)) / scatterPart),
                        };
                    }
                }

                break;
            case (DistributedType a, DistributedType b) when a.TensorType != b.TensorType && a.Placement == b.Placement:
                cost = new Cost()
                {
                    [CostFactorNames.MemoryStore] = (UInt128)((float)CostUtility.GetMemoryAccess(a) / DistributedUtility.GetDividedTensorEfficiency(a, _burstLength)),
                    [CostFactorNames.MemoryLoad] = (UInt128)((float)CostUtility.GetMemoryAccess(b) / DistributedUtility.GetDividedTensorEfficiency(b, _burstLength)),
                };
                break;
            case (DistributedType a, DistributedType b) when a == b:
                throw new InvalidOperationException($"the boxing inType == outType");
            default:
                throw new NotSupportedException($"{inType} {returnType}");
        }

        return cost;
    }

    public IValue Visit(IEvaluateContext context, Boxing target)
    {
        return context.GetArgumentValue(target, Boxing.Input);
    }
}
