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
[EvaluatorGenerator, TypeInferGenerator]
public partial class GetItemEvaluator : IEvaluator<GetItem>, ITypeInferencer<GetItem>
{
    private byte[] ObjectToByteArray(object obj)
    {
        if (obj == null)
            return new byte[] { };
        BinaryFormatter bf = new BinaryFormatter();
        MemoryStream ms = new MemoryStream();
        bf.Serialize(ms, obj);
        return ms.ToArray();
    }

    Tensor Visit(IValue Input, int Index)
    {
        if (Input.Type is TensorType ttype)
        {
            return Tensor.FromBytes(TensorType.Scalar(ttype.DType), ObjectToByteArray(Input.AsTensor()[Index]));
        }
        return Input.AsTensors()[Index];
    }

    IRType Visit(ITypeInferenceContext context, GetItem target, IRType Input)
    {
        IRType ret = new InvalidType("Need Be Reset!");
        switch (Input)
        {
            case TensorType tensorType:
                ret = TensorType.Scalar(tensorType.DType);
                break;
            case TupleType tupleType:
                if (context.GetArgument(target, GetItem.Index) is TensorConst @const)
                {
                    var index = @const.Value.ToScalar<int>();
                    if (index < tupleType.Count)
                        ret = tupleType[index];
                    else
                        ret = new InvalidType($"The Input Tuple Count = {tupleType.Count}, But Index = {index}");
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
