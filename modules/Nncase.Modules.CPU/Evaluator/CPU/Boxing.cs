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

namespace Nncase.Evaluator.IR.CPU;

public sealed class BoxingEvaluator : ITypeInferencer<Boxing>, ICostEvaluator<Boxing>, IEvaluator<Boxing>
{
    public static IRType VisitType(IRType inType, IRType outType, bool isReshape = false)
    {
        IRType VisitD2D(DistributedType inv, DistributedType outv)
        {
            if (inv.TensorType != outv.TensorType)
            {
                if (!inv.NdSBP.Any(sbp => sbp is SBPPartial))
                {
                    return outv;
                }
            }

            // TODO: add more invalid cases
            if (inv.NdSBP.Distinct().Count() == 1 && outv.NdSBP.Distinct().Count() == 1 && inv.NdSBP[0] == outv.NdSBP[0])
            {
                return new InvalidType("Same NDSBP");
            }

            for (int i = 0; i < inv.NdSBP.Count; i++)
            {
                switch (inv.NdSBP[i], outv.NdSBP[i])
                {
                    case (SBPPartial, SBPSplit):
                        return new InvalidType("partial to split");
                    case (not SBPPartial, SBPPartial):
                        return new InvalidType("split/broadcast to partial");
                }
            }

            return outv;
        }

        IRType VisitD2T(DistributedType inv, TensorType outv)
        {
            if (inv.NdSBP.Any(s => s is SBPPartial))
            {
                return new InvalidType("Not supported input is Partial output is Unshard");
            }

            return outv;
        }

        IRType VisitT2D(TensorType inv, DistributedType outv)
        {
            if (outv.NdSBP.Any(s => s is SBPPartial))
            {
                return new InvalidType("Not supported input is Unshard output is Partial");
            }

            return outv;
        }

        return (inType, outType) switch
        {
            (InvalidType inv, _) => inv,
            (_, InvalidType inv) => inv,
            (DistributedType d, DistributedType d1) => VisitD2D(d, d1),
            (TensorType t, DistributedType d) => VisitT2D(t, d),
            (DistributedType d, TensorType t) => VisitD2T(d, t),
            _ => new InvalidType($"not support boxing {inType} to {outType}"),
        };
    }

    public IRType Visit(ITypeInferenceContext context, Boxing target)
    {
        return VisitType(context.GetArgumentType(target, Boxing.Input), target.NewType);
    }

    public Cost Visit(ICostEvaluateContext context, Boxing target)
    {
        var inType = context.GetArgumentType<IRType>(target, Boxing.Input);
        var returnType = context.GetReturnType<IRType>();
        var cost = new Cost() { [CostFactorNames.MemoryLoad] = 0, [CostFactorNames.MemoryStore] = 0 };
        switch (inType, returnType)
        {
            case (TensorType _, DistributedType distTensorType):
                switch (context.CompileOptions.TargetOptions)
                {
                    default:
                        cost = new Cost()
                        {
                            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(distTensorType),
                        };
                        break;
                }

                break;
            case (DistributedType distTensorType, TensorType _):
                switch (context.CompileOptions.TargetOptions)
                {
                    default:
                        cost = new Cost()
                        {
                            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(distTensorType),
                        };
                        break;
                }

                break;

            case (DistributedType a, DistributedType b) when a.Placement == b.Placement && a.NdSBP != b.NdSBP:
                {
                    var fullLoadStore = new Cost()
                    {
                        [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(a),
                        [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(b),
                    };

                    float scatterPart = 1;
                    float gatherPart = 1;
                    float reducePart = 1;
                    float latency = 0;
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
                                cost += new Cost()
                                {
                                    [CostFactorNames.CPUCycles] = 1,
                                };
                                break;
                            case (SBPPartial, SBP sbpout):
                                switch (sbpout)
                                {
                                    case SBPPartial:
                                        break;
                                    case SBPBroadCast:
                                        latency = MathF.Max(latency, ((ICpuTargetOptions)context.CompileOptions.TargetOptions).HierarchyLatencies[i]);
                                        reducePart *= a.Placement.Hierarchy[i];
                                        gatherPart *= a.Placement.Hierarchy[i];
                                        if (i == 0)
                                        {
                                            scatterPart *= a.Placement.Hierarchy[i];
                                        }

                                        break;
                                    case SBPSplit:
                                        throw new NotSupportedException("split to partial");
                                }

                                break;
                            case (SBPBroadCast, SBPPartial):
                                // note this case only for tests.
                                cost += new Cost()
                                {
                                    [CostFactorNames.CPUCycles] = 1,
                                };
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

                        if (a.Placement.HierarchyKind == HierarchyKind.SMT && (a.NdSBP[1] is SBPPartial))
                        {
                            cost[CostFactorNames.MemoryStore] *= 8;
                        }
                    }

                    if (reducePart > 1f)
                    {
                        cost += new Cost()
                        {
                            [CostFactorNames.Comm] = (UInt128)((reducePart - 1) * latency),
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
                    [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(a),
                    [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(b),
                };
                break;
            case (DistributedType a, DistributedType b) when a.Placement != b.Placement:
                cost = new Cost()
                {
                    [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(a),
                    [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(b),
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
        var input = context.GetArgumentValueAsTensor(target, Boxing.Input);
        return target.NewType switch
        {
            TensorType t => Value.FromTensor(Tensor.FromBytes(input.ElementType, input.BytesBuffer.ToArray(), t.Shape)),
            DistributedType d => Value.FromTensor(Tensor.FromBytes(input.ElementType, input.BytesBuffer.ToArray(), d.TensorType.Shape), d.NdSBP, d.Placement),
            _ => Value.FromTensor(input),
        };
    }
}
