// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="GetItem"/>.
/// </summary>
[EvaluatorGenerator]
[TypeInferGenerator]
public partial class GetItemEvaluator : IEvaluator<GetItem>, ITypeInferencer<GetItem>
{
    private Tensor Visit(IValue Input, int Index)
    {
        if (Input.Type is TensorType ttype)
        {
            var tensor = Input.AsTensor();
            if (tensor.Rank != 1)
            {
                throw new InvalidOperationException();
            }

            var elementSize = tensor.ElementType.SizeInBytes;
            var src = tensor.BytesBuffer.Slice(elementSize * Index, elementSize);
            return Tensor.FromBytes(TensorType.Scalar(ttype.DType), src);
        }

        return Input.AsTensors()[Index];
    }

    private IRType Visit(ITypeInferenceContext context, GetItem target, IRType Input)
    {
        IRType ret = new InvalidType("Need Be Reset!");
        switch (Input)
        {
            case TensorType tensorType:
                if (tensorType.Shape.Rank != 1)
                {
                    ret = new InvalidType($"The Input tensor's rank should be 1, but {tensorType.Shape.Rank}");
                }
                else
                {
                    ret = TensorType.Scalar(tensorType.DType);
                }

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
