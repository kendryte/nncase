// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#pragma warning disable SA1008 // Opening parenthesis should be spaced correctly

using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.NTT;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// Evaluator for <see cref="VectorizedCast"/>.
/// </summary>
public class VectorizedCastEvaluator : IEvaluator<VectorizedCast>, ITypeInferencer<VectorizedCast>, IOpPrinter<VectorizedCast>, ICostEvaluator<VectorizedCast>, IMetricEvaluator<VectorizedCast>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, VectorizedCast cast)
    {
        var input = context.GetArgumentValue(cast, VectorizedCast.Input).AsTensor();
        IValue result;
        if (cast.NewType is VectorType vt && !cast.VectorizeAxes.IsDefaultOrEmpty)
        {
            if (cast.VectorizeAxes.Count > 1)
            {
                throw new NotSupportedException("Vectorize axes must be one");
            }

            input = Nncase.IR.F.Tensors.Unpack(input, ((VectorType)input.ElementType).Lanes.ToArray(), cast.VectorizeAxes.ToArray()).Evaluate().AsTensor();
            input = input.CastTo(vt.ElemType);
            input = Nncase.IR.F.Tensors.Pack(input, vt.Lanes.ToArray(), cast.VectorizeAxes.ToArray()).Evaluate().AsTensor();
            result = Value.FromTensor(input);
        }
        else
        {
            result = Value.FromTensor(input.CastTo(cast.NewType, cast.CastMode));
        }

        if (context.CurrentCall[VectorizedCast.PostOps] is Fusion lambda)
        {
            return CompilerServices.Evaluate(lambda.Body, new Dictionary<IVar, IValue>() { { lambda.Parameters[0], result } });
        }

        return result;
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, VectorizedCast target)
    {
        var input = context.CheckArgumentType<IRType>(target, VectorizedCast.Input);
        var postOps = context.CheckArgumentType<IRType>(target, VectorizedCast.PostOps);
        if (!(postOps is NoneType || postOps is CallableType))
        {
            return new InvalidType($"PostOps must be None or Callable, but got {postOps}");
        }

        return input switch
        {
            TensorType t => Visit(target, t),
            DistributedType d => Visit(target, d),
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public string Visit(IPrintOpContext context, VectorizedCast target)
    {
        return $"{CompilerServices.Print(target.NewType)}({context.GetArgument(target, VectorizedCast.Input)})";
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, VectorizedCast target)
    {
        var input = context.GetArgumentType<IRType>(target, VectorizedCast.Input);
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(target.NewType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(target.NewType, 1),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, VectorizedCast target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, VectorizedCast.Input);
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) * 2,
        };
    }

    private IRType Visit(VectorizedCast target, TensorType input)
    {
        if (input.DType is VectorType vt)
        {
            if (!target.VectorizeAxes.IsDefaultOrEmpty && target.VectorizeAxes.Any(a => input.Shape[a] is { IsFixed: false }))
            {
                return new InvalidType("Vectorize axes must be fixed");
            }

            var scale = 1f;
            var newShape = input.Shape.ToArray();
            if (!target.VectorizeAxes.IsDefaultOrEmpty)
            {
                scale = 1f * ((VectorType)target.NewType).ElemType.SizeInBytes / vt.ElemType.SizeInBytes;
                if (target.VectorizeAxes.Any(a => input.Shape[a].FixedValue * scale % 1 != 0))
                {
                    return new InvalidType("Vectorize axes must be divisible by scale");
                }

                foreach (var a in target.VectorizeAxes)
                {
                    newShape[a] = (int)(newShape[a].FixedValue * scale);
                }
            }

            return new TensorType(target.NewType, newShape);
        }

        return new TensorType(target.NewType, input.Shape);
    }

    private IRType Visit(VectorizedCast target, DistributedType inType)
    {
        var invalid = new InvalidType(inType.ToString());
        var outType = Visit(target, inType.TensorType);
        var ndsbp = new SBP[inType.TensorType.Shape.Rank];
        var shape = CompilerServices.GetMaxShape(inType.TensorType.Shape);
        for (int i = 0; i < ndsbp.Length; i++)
        {
            if (inType.AxisPolicies[i] is SBPPartial)
            {
                return invalid;
            }

            if (inType.AxisPolicies[i] is SBPSplit split && inType.TensorType.DType is VectorType vtIn && outType is TensorType ttOut && ttOut.DType is VectorType vtOut)
            {
                var outShape = CompilerServices.GetMaxShape(ttOut.Shape);
                if (vtIn.ElemType != vtOut.ElemType)
                {
                    var divisor = split.Axes.Select(a => inType.Placement.Hierarchy[a]).Aggregate(1, (a, b) => a * b);
                    if (shape[i] % divisor != 0 || outShape[i] % divisor != 0)
                    {
                        return invalid;
                    }
                }
            }

            ndsbp[i] = inType.AxisPolicies[i];
        }

        return new DistributedType((TensorType)outType, ndsbp, inType.Placement);
    }
}
