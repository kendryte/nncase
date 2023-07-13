// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using Tuple = System.Tuple;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="GetItem"/>.
/// </summary>
[EvaluatorGenerator]
[TypeInferGenerator]
public partial class GetItemEvaluator : IEvaluator<GetItem>, ITypeInferencer<GetItem>, IOpPrinter<GetItem>, ICostEvaluator<GetItem>, IShapeEvaluator<GetItem>
{
    public string Visit(IIRPrinterContext context, GetItem target, bool iLmode)
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

    public Expr Visit(IShapeEvaluateContext context, GetItem target)
    {
        var input = context.GetArgumentShape(target, GetItem.Input);
        var index = context.GetArgument(target, GetItem.Index);
        if (input is IR.Tuple)
        {
            return input[index];
        }
        else
        {
            return IR.F.Tensors.ShapeOf(input[index]);
        }
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
            var returnDims = tensor.Dimensions.AsValueEnumerable().Skip(indexTensor.Length).ToArray();
            var elementsCount = (int)TensorUtilities.GetProduct(returnDims);

            var src = tensor.BytesBuffer.Slice(elementSize * linearIndex, elementSize * elementsCount);
            return Value.FromTensor(Tensor.FromBytes(new TensorType(ttype.DType, returnDims), src.ToArray()));
        }

        return input[index.AsTensor().ToScalar<int>()];
    }

    private IRType Visit(ITypeInferenceContext context, GetItem target, IRType input, TensorType index)
    {
        IRType ret = new InvalidType("Need Be Reset!");
        switch (input)
        {
            case TensorType tensorType:
                if (tensorType.Shape.IsUnranked)
                {
                    return input;
                }

                var shape = index.Shape switch
                {
                    { IsScalar: true } => new Shape(tensorType.Shape.Skip(1)),
                    { IsFixed: true } => index.Shape[0].FixedValue == tensorType.Shape.Rank ?
                                         Shape.Scalar :
                                         new Shape(tensorType.Shape.Skip(index.Shape[0].FixedValue)),
                    _ => Shape.Unranked,
                };
                ret = new TensorType(tensorType.DType, shape);
                break;
            case TupleType tupleType:
                if (context.GetArgument(target, GetItem.Index) is TensorConst @const)
                {
                    var indexValue = @const.Value.ToScalar<int>();
                    if (indexValue < tupleType.Count)
                    {
                        ret = tupleType[indexValue];
                    }
                    else
                    {
                        if (tupleType.IsVariadic)
                        {
                            ret = tupleType[0];
                        }
                        else
                        {
                            ret = new InvalidType($"The Input Tuple Count = {tupleType.Count}, But Index = {indexValue}");
                        }
                    }
                }
                else
                {
                    ret = AnyType.Default;
                }

                break;
            default:
                break;
        }

        return ret;
    }
}
