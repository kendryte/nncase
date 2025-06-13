// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using Tuple = System.Tuple;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="GetItem"/>.
/// </summary>
[EvaluatorGenerator]
[TypeInferGenerator]
public partial class GetItemEvaluator : IEvaluator<GetItem>, ITypeInferencer<GetItem>, IOpPrinter<GetItem>, ICostEvaluator<GetItem>, IMetricEvaluator<GetItem>
{
    public string Visit(IPrintOpContext context, GetItem target)
    {
        return $"{context.GetArgument(target, GetItem.Input)}[{context.GetArgument(target, GetItem.Index)}]";
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, GetItem target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    public Metric Visit(IMetricEvaluateContext context, GetItem target)
    {
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public IRType Visit(ITypeInferenceContext context, GetItem target)
    {
        var input = context.CheckArgumentType<IRType>(target, GetItem.Input);
        var index = context.CheckArgumentType<IRType>(target, GetItem.Index);

        return input switch
        {
            TensorType t => Visit(context, target, t, index),
            DistributedType d => Visit(context, target, d, index),
            TupleType t => Visit(context, target, t, index),
            AnyType => AnyType.Default,
            _ => new InvalidType(input.GetType().Name),
        };
    }

    private IValue Visit(IValue input, IValue index)
    {
        if (input.Type is TensorType ttype)
        {
            var tensor = input.AsTensor();
            var elementSize = tensor.ElementType.SizeInBytes;
            var indices = new int[tensor.Rank];
            var indexTensor = index.AsTensor().Cast<int>();
            indexTensor.Buffer.CopyTo(indices);
            var indicesValue = indices.Select((x, i) => x < 0 ? x + tensor.Shape[i].FixedValue : x).ToArray();
            var linearIndex =
                TensorUtilities.GetIndex(tensor.Strides, indicesValue);
            var returnDims = tensor.Dimensions.AsValueEnumerable().Skip((int)indexTensor.Length).ToArray();
            var elementsCount = TensorUtilities.GetProduct(returnDims);

            var src = tensor.BytesBuffer.Slice(checked((int)(elementSize * linearIndex)), checked((int)(elementSize * elementsCount)));
            return Value.FromTensor(Tensor.FromBytes(new TensorType(ttype.DType, returnDims), src.ToArray()));
        }

        return input[index.AsTensor().ToScalar<int>()];
    }

    private IRType Visit(ITypeInferenceContext context, GetItem target, TensorType input, IRType index)
    {
        var indexExpr = context.GetArgument(target, GetItem.Index);
        if (input.Shape is not RankedShape inShape)
        {
            return input;
        }

        var indices = indexExpr switch
        {
            DimConst dc => new[] { dc.Value },
            RankedShape rs when rs.IsFixed => rs.ToValueArray(),
            _ => null,
        };
        if (indices is not null)
        {
            if (indices.Length > inShape.Rank)
            {
                return new InvalidType("GetItem index count should smaller than in shape rank");
            }

            if (indices.Length == inShape.Rank)
            {
                foreach (var (i, dim) in indices.Zip(inShape))
                {
                    if (dim.IsFixed && i >= dim.FixedValue)
                    {
                        return new InvalidType("GetItem index value shoud smaller than shape dim");
                    }
                }
            }
        }

        Shape shape = index switch
        {
            DimensionType => new RankedShape(inShape.Skip(1)),
            ShapeType st when st.Rank != null => st.Rank == inShape.Rank ?
                                 Shape.Scalar :
                                 new RankedShape(inShape.Skip(st.Rank.Value)),
            _ => Shape.Unranked,
        };
        return new TensorType(input.DType, shape);
    }

    private IRType Visit(ITypeInferenceContext context, GetItem target, TupleType input, IRType index)
    {
        var indexExpr = context.GetArgument(target, GetItem.Index);
        if (indexExpr is DimConst @const)
        {
            var indexValue = (int)@const.Value;
            if (indexValue < 0)
            {
                return new InvalidType($"The Input Tuple Count = {input.Count}, But Index = {indexValue}");
            }
            else if (indexValue < input.Count)
            {
                return input[indexValue];
            }
            else
            {
                if (input.IsVariadic)
                {
                    return input[0];
                }
                else
                {
                    return new InvalidType($"The Input Tuple Count = {input.Count}, But Index = {indexValue}");
                }
            }
        }
        else
        {
            return AnyType.Default;
        }
    }

    private IRType Visit(ITypeInferenceContext context, GetItem target, DistributedType input, IRType index)
    {
        var outputType = (TensorType)Visit(context, target, input.TensorType, index);
        var ndsbp = input.AxisPolicies.Skip(input.TensorType.Shape.Rank - outputType.Shape.Rank).ToArray();
        for (var i = 0; i < ndsbp.Length; i++)
        {
            if (ndsbp[i] is SBPSplit)
            {
                if ((index is DimensionType && i == 0)
                    || i < ((ShapeType)index).Rank)
                {
                    return new InvalidType("not support");
                }
            }
        }

        return new DistributedType(outputType, ndsbp, input.Placement);
    }
}
