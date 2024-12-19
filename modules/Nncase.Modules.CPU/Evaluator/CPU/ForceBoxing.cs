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

public sealed class ForceBoxingEvaluator : ITypeInferencer<ForceBoxing>, ICostEvaluator<ForceBoxing>, IEvaluator<ForceBoxing>
{
    public static IRType VisitType(IRType inType, IRType outType)
    {
        IRType VisitD2D(DistributedType inv, DistributedType outv)
        {
            // TODO: add more invalid cases
            if (inv.NdSBP.Distinct().Count() == 1 && outv.NdSBP.Distinct().Count() == 1 && inv.NdSBP[0] == outv.NdSBP[0])
            {
                return new InvalidType("Same NDSBP");
            }

            if (inv.NdSBP.Any(sbp => sbp is SBPPartial))
            {
                var nonPartialSumPos = Enumerable.Range(0, inv.NdSBP.Count).Where(i => inv.NdSBP[i] is not SBPPartial);
                if (nonPartialSumPos.Any(i => inv.NdSBP[i] is SBPSplit && outv.NdSBP[i] is SBPBroadCast))
                {
                    return new InvalidType("Not supported input is Split output is BroadCast");
                }

                var partialSumPos = Enumerable.Range(0, inv.NdSBP.Count).Where(i => inv.NdSBP[i] is SBPPartial);
                if (partialSumPos.Any(i => inv.NdSBP[i] is SBPPartial && outv.NdSBP[i] is SBPSplit))
                {
                    return new InvalidType("Not supported input is Partial output is Split");
                }

                return outv;
            }

            if (outv.NdSBP.Any(sbp => sbp is SBPPartial))
            {
                var nonPartialSumPos = Enumerable.Range(0, outv.NdSBP.Count).Where(i => outv.NdSBP[i] is not SBPPartial);
                if (nonPartialSumPos.Any(i => inv.NdSBP[i] is SBPSplit && outv.NdSBP[i] is SBPBroadCast))
                {
                    return new InvalidType("Not supported input is Split output is BroadCast");
                }

                var partialSumPos = Enumerable.Range(0, outv.NdSBP.Count).Where(i => outv.NdSBP[i] is SBPPartial);
                if (partialSumPos.Any(i => inv.NdSBP[i] is SBPSplit && outv.NdSBP[i] is SBPPartial))
                {
                    return new InvalidType("Not supported input is Split output is Partial");
                }

                return outv;
            }

            return outv;
        }

        return (inType, outType) switch
        {
            (InvalidType inv, _) => inv,
            (_, InvalidType inv) => inv,
            (DistributedType d, DistributedType d1) => VisitD2D(d, d1),
            _ => new InvalidType($"not support Forceboxing {inType} to {outType}"),
        };
    }

    public IRType Visit(ITypeInferenceContext context, ForceBoxing target)
    {
        return VisitType(context.GetArgumentType(target, ForceBoxing.Input), target.NewType);
    }

    public Cost Visit(ICostEvaluateContext context, ForceBoxing target)
    {
        var inType = context.GetArgumentType<IRType>(target, ForceBoxing.Input);
        var returnType = context.GetReturnType<IRType>();
        var cost = new Cost() { [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inType), [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType) };
        return cost;
    }

    public IValue Visit(IEvaluateContext context, ForceBoxing target)
    {
        var inTenor = context.GetArgumentValueAsTensor(target, ForceBoxing.Input);
        var input = inTenor.ToOrtTensor();
        var output = input - input;
        var repeat = target.NewType.NdSBP.Select((x, i) => (x is SBPPartial) ? target.NewType.Placement.Hierarchy[i] : 1).Aggregate(1, (x, i) => x * i);
        for (int i = 0; i < repeat; i++)
        {
            output += input;
        }

        return Value.FromTensor(Tensor.FromBytes(inTenor.ElementType, output.BytesBuffer.ToArray(), target.NewType.TensorType.Shape));
    }
}
