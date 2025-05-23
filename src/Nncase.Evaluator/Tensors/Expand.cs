// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using OrtKISharp;
using Shape = Nncase.IR.Shape;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Expand"/>.
/// </summary>
[TypeInferGenerator]
public sealed partial class ExpandEvaluator : IEvaluator<Expand>, ITypeInferencer<Expand>, ICostEvaluator<Expand>, IMetricEvaluator<Expand>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Expand expand)
    {
        var input = context.GetArgumentValueAsTensor(expand, Expand.Input);
        var originType = input.ElementType;
        if (originType is VectorType { ElemType: DataType dataTypes } vType && dataTypes != DataTypes.Float32)
        {
            var interType = new VectorType(DataTypes.Float32, vType.Lanes);
            input = input.CastTo(interType);
        }
        else if (originType.IsFloat() && originType is not VectorType && originType != DataTypes.Float32)
        {
            input = input.CastTo(DataTypes.Float32);
        }

        var inputOrt = input.ToOrtTensor();
        var shape = context.GetArgumentValue(expand, Expand.Shape).AsTensor().ToArray<long>();
        if (originType is VectorType)
        {
            shape = shape.Concat(((VectorType)input.ElementType).Lanes.Select(lane => (long)lane)).ToArray();
        }

        return OrtKI.Expand(inputOrt, Tensor.FromArray(shape).ToOrtTensor()).ToValue(originType);
    }

    public Cost Visit(ICostEvaluateContext context, Expand target)
    {
        var input = context.GetArgumentType<IRType>(target, Expand.Input);
        var ret = context.GetReturnType<IRType>();

        return CostUtility.GetBroadcastCost(input, ret);
    }

    public Metric Visit(IMetricEvaluateContext context, Expand target)
    {
        var input = context.GetArgumentType<TensorType>(target, Expand.Input);
        var ret = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(input) + CostUtility.GetMemoryAccess(ret),
        };
    }

    public IRType Visit(ITypeInferenceContext context, Expand target)
    {
        var input = context.CheckArgumentType<IRType>(target, Expand.Input);
        var shape = (Shape)context.GetArgument(target, Expand.Shape);
        return input switch
        {
            TensorType t => Visit(context, target, t, shape),
            DistributedType d => Visit(context, target, d, shape),
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    private IRType Visit(ITypeInferenceContext context, Expand target, TensorType input, Shape shape)
    {
        var newShape = TypeInference.ExpandShape(input.Shape, shape);
        return input with { Shape = newShape };
    }

    private IRType Visit(ITypeInferenceContext context, Expand target, DistributedType input, Shape shape)
    {
        var invalid = new InvalidType(input.ToString());
        if (input.TensorType.Shape.IsRanked && shape.IsRanked)
        {
            var newShape = TypeInference.ExpandShape(input.TensorType.Shape, shape);
            var dimExtends = newShape.Rank - input.TensorType.Shape.Rank;
            var ndsbp = new SBP[newShape.Rank];
            for (int i = 0; i < ndsbp.Length; i++)
            {
                if (i < dimExtends)
                {
                    ndsbp[i] = SBP.B;
                }
                else
                {
                    if (input.AxisPolices[i - dimExtends] is SBPSplit && newShape[i] != input.TensorType.Shape[i - dimExtends])
                    {
                        return invalid;
                    }

                    ndsbp[i] = input.AxisPolices[i - dimExtends];
                }
            }

            return new DistributedType(new TensorType(input.TensorType.DType, newShape), ndsbp, input.Placement);
        }

        return invalid;
    }
}
