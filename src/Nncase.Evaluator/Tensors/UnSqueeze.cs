﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Unsqueeze"/>.
/// </summary>
public class UnsqueezeEvaluator : IEvaluator<Unsqueeze>, ITypeInferencer<Unsqueeze>, ICostEvaluator<Unsqueeze>, IMetricEvaluator<Unsqueeze>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Unsqueeze unSqueeze)
    {
        OrtKISharp.Tensor input;
        var inputOrg = context.GetArgumentValue(unSqueeze, Unsqueeze.Input).AsTensor();
        var dataType = inputOrg.ElementType;
        if (dataType is VectorType { ElemType: DataType dataTypes } vType && dataTypes != DataTypes.Float32)
        {
            var interType = new VectorType(DataTypes.Float32, vType.Lanes);
            input = Nncase.IR.F.Tensors.Cast(inputOrg, interType).Evaluate().AsTensor().ToOrtTensor();
        }
        else if (dataType is not VectorType && dataType.IsFloat() && dataType != DataTypes.Float32)
        {
            input = Nncase.IR.F.Tensors.Cast(inputOrg, DataTypes.Float32).Evaluate().AsTensor().ToOrtTensor();
        }
        else
        {
            input = context.GetOrtArgumentValue(unSqueeze, Unsqueeze.Input);
        }

        var axes = context.GetInt64OrtTensorArgumentValue(unSqueeze, Unsqueeze.Dim);
        if (dataType.IsFloat() && dataType != DataTypes.Float32)
        {
            var unsequeeze = OrtKI.Unsqueeze(input, axes);
            if (dataType is not VectorType && dataType != DataTypes.Float32)
            {
                unsequeeze = OrtKI.Cast(unsequeeze, (int)dataType.ToOrtType());
            }

            return Value.FromTensor(unsequeeze.ToTensor(context.CurrentCall.CheckedTensorType).CastTo(dataType));
        }
        else
        {
            return Value.FromTensor(OrtKI.Unsqueeze(input, axes).ToTensor(context.CurrentCall.CheckedTensorType));
        }
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Unsqueeze target)
    {
        var input = context.CheckArgumentType<IRType>(target, Unsqueeze.Input);
        if (input is TensorType tensorType)
        {
            return Visit(context, target, tensorType);
        }
        else if (input is DistributedType distributedType)
        {
            return Visit(context, target, distributedType);
        }

        return new InvalidType(input.GetType().Name);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Unsqueeze target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Unsqueeze target) => Metric.Zero;

    private IRType Visit(ITypeInferenceContext context, Unsqueeze target, TensorType input)
    {
        if (input.Shape.IsUnranked)
        {
            return input;
        }

        if (context.GetArgument(target, Unsqueeze.Dim) is RankedShape { IsFixed: true } axes)
        {
            var axesValue = axes.ToValueArray();
            var outShape = new Dimension[input.Shape.Rank + axesValue.Length];
            axesValue = axesValue.Select(axis => axis < 0 ? axis + outShape.Length : axis).ToArray();
            var offset = 0;
            for (int i = 0; i < outShape.Length; i++)
            {
                if (axesValue.Contains(i))
                {
                    outShape[i] = 1;
                }
                else
                {
                    outShape[i] = input.Shape[offset++];
                }
            }

            return input with { Shape = new RankedShape(outShape) };
        }

        return input with { Shape = Shape.Unknown(input.Shape.Rank + 1) };
    }

    private IRType Visit(ITypeInferenceContext context, Unsqueeze target, DistributedType input)
    {
        var tensorType = (TensorType)Visit(context, target, input.TensorType);

        var ndsbp = Enumerable.Repeat(SBP.B, tensorType.Shape.Rank).Select(b => (SBP)b).ToArray();
        var dim = (RankedShape)context.GetArgument(target, Unsqueeze.Dim);

        if (dim.IsFixed)
        {
            var dimsValue = dim.ToValueArray().Select(d => d < 0 ? d + tensorType.Shape.Rank : d);
            for (int i = 0; i < input.AxisPolices.Count; i++)
            {
                var outAxis = i + dimsValue.Select(d => d <= i).Count(b => b);
                if (dimsValue.Contains(outAxis))
                {
                    ndsbp[outAxis] = SBP.B;
                }
                else
                {
                    ndsbp[outAxis] = input.AxisPolices[i];
                }
            }
        }

        return new DistributedType(tensorType, ndsbp, input.Placement);
    }
}
