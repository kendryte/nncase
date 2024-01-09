// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Ncnn;
using Nncase.Utilities;
using static Nncase.IR.F.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnConcat"/>.
/// </summary>
public class NcnnConcatEvaluator : IEvaluator<NcnnConcat>, ITypeInferencer<NcnnConcat>, ICostEvaluator<NcnnConcat>, IShapeEvaluator<NcnnConcat>, IMetricEvaluator<NcnnConcat>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnConcat concat)
    {
        var inputs = context.GetArgumentValueAsTensors(concat, NcnnConcat.Input);
        var axis = concat.Axis;
        return OrtKI.Concat(inputs.Select(t => t.ToOrtTensor()).ToArray(), axis).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnConcat target)
    {
        var inputs = context.CheckArgumentType<TupleType>(target, NcnnConcat.Input);
        return Visit(inputs.Fields, target.Axis);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnConcat target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnConcat target) => Metric.Zero;

    public Expr Visit(IShapeEvaluateContext context, NcnnConcat target) => context.GetArgumentShape(target, NcnnConcat.Input);

    private IRType? CheckType(TupleType inputs)
    {
        bool? allScalar = null;
        DataType? allDType = null;
        foreach (var (i, input) in Enumerable.Range(0, inputs.Count).Select(i => (i, inputs[i])))
        {
            TensorType type;
            if (input is TensorType a)
            {
                type = a;
            }
            else if (input is DistributedType { TensorType: TensorType b })
            {
                type = b;
            }
            else
            {
                return new InvalidType($"The ConCat Item[{i}] Must Have TensorType But Get {input}");
            }

            if (type.Shape.IsUnranked)
            {
                return new TensorType(type.DType, Shape.Unranked);
            }

            allScalar = (allScalar ?? type.IsScalar) & type.IsScalar;
            allDType ??= type.DType;
            if (allDType != type.DType)
            {
                return new InvalidType(
                    $"The ConCat Item[{i}] Must Be {allDType} But Get {type.DType.GetDisplayName()}");
            }
        }

        if (allScalar == true && allDType is not null)
        {
            return new TensorType(allDType, new[] { inputs.Count });
        }

        return null;
    }

    private TensorType GetTensorType(IRType input) => input switch
    {
        TensorType t => t,
        DistributedType d => d.TensorType,
        _ => throw new InvalidCastException(),
    };


    private IRType Visit(IRArray<IRType> inputs, int axisValue)
    {
        var outputShape = GetTensorType(inputs[0]).Shape.ToArray();

        foreach (var item in inputs[1..])
        {
            outputShape[axisValue] += GetTensorType(item).Shape.ToArray()[axisValue];
        }

        return new TensorType(GetTensorType(inputs[0]).DType, outputShape);
    }
}
