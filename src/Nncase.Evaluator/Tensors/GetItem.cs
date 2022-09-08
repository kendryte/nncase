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

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="GetItem"/>.
/// </summary>
[EvaluatorGenerator]
[TypeInferGenerator]
public partial class GetItemEvaluator : IEvaluator<GetItem>, ITypeInferencer<GetItem>, IOpPrinter<GetItem>, ICostEvaluator<GetItem>
{
    public string Visit(IIRPrinterContext context, GetItem target, bool ILmode)
    {
        return $"{context.GetArgument(target, GetItem.Input)}[{context.GetArgument(target, GetItem.Index)}]";
    }

    /// <inheritdoc/>
    public Cost? Visit(ICostEvaluateContext context, GetItem target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    IValue Visit(IValue Input, IValue Index)
    {
        if (Input.Type is TensorType ttype)
        {
            var tensor = Input.AsTensor();
            var elementSize = tensor.ElementType.SizeInBytes;
            var indices = new int[tensor.Rank];
            var indexTensor = Index.AsTensor().Cast<int>();
            indexTensor.Buffer.CopyTo(indices);
            var indicesValue = indices.Select((x, i) => (x < 0 ? x + tensor.Shape[i] : x).FixedValue).ToArray();
            var linearIndex =
                TensorUtilities.GetIndex(tensor.Strides, indicesValue);
            var returnDims = tensor.Dimensions.AsValueEnumerable().Skip(indexTensor.Length).ToArray();
            var elementsCount = (int)TensorUtilities.GetProduct(returnDims);

            var src = tensor.BytesBuffer.Slice(elementSize * linearIndex, elementSize * elementsCount);
            return Value.FromTensor(Tensor.FromBytes(new TensorType(ttype.DType, returnDims), src));
        }

        return Input[Index.AsTensor().ToScalar<int>()];
    }

    private IRType Visit(ITypeInferenceContext context, GetItem target, IRType Input, TensorType Index)
    {
        IRType ret = new InvalidType("Need Be Reset!");
        switch (Input)
        {
            case TensorType tensorType:
                if (tensorType.Shape.IsUnranked)
                {
                    return Input;
                }
                ret = new TensorType(tensorType.DType,
                       Index.Shape switch
                       {
                           { IsScalar: true } => new Shape(tensorType.Shape.Skip(1)),
                           { IsFixed: true } => Index.Shape[0].FixedValue == tensorType.Shape.Rank ? 
                                                Shape.Scalar :
                                                new Shape(tensorType.Shape.Skip(Index.Shape[0].FixedValue)),
                           _ => Shape.Unranked,
                       });
                break;
            case TupleType tupleType:
                if (context.GetArgument(target, GetItem.Index) is TensorConst @const)
                {
                    var index = @const.Value.ToScalar<int>();
                    if (index < tupleType.Count)
                    {
                        ret = tupleType[index];
                    }
                    else
                    {
                        ret = new InvalidType($"The Input Tuple Count = {tupleType.Count}, But Index = {index}");
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
