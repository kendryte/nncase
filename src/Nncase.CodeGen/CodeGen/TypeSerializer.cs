// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Runtime;

namespace Nncase.CodeGen;

/// <summary>
/// Type serializer.
/// </summary>
public static class TypeSerializer
{
    public static void Serialize(BinaryWriter writer, DataType dataType)
    {
        switch (dataType)
        {
            case PrimType t:
                writer.Write((byte)t.TypeCode);
                break;
            case PointerType t:
                writer.Write((byte)Runtime.TypeCode.Pointer);
                Serialize(writer, t.ElemType);
                break;
            case ValueType t:
                writer.Write((byte)Runtime.TypeCode.ValueType);
                writer.Write(t.Uuid.ToByteArray());
                writer.Write(t.SizeInBytes);
                break;
            case VectorType t:
                writer.Write((byte)Runtime.TypeCode.VectorType);
                Serialize(writer, t.ElemType);
                writer.Write(checked((byte)t.Lanes.Count));
                for (int i = 0; i < t.Lanes.Count; i++)
                {
                    writer.Write(checked((byte)t.Lanes[i]));
                }

                break;
            default:
                throw new ArgumentException($"Unsupported datatype: {dataType}");
        }
    }

    public static void Serialize(BinaryWriter writer, IRType type)
    {
        switch (type)
        {
            case InvalidType:
                writer.Write((byte)TypeSignatureToken.Invalid);
                break;
            case AnyType:
                writer.Write((byte)TypeSignatureToken.Any);
                break;
            case TensorType t:
                writer.Write((byte)TypeSignatureToken.Tensor);
                Serialize(writer, t.DType);
                Serialize(writer, t.Shape);
                break;
            case TupleType t:
                writer.Write((byte)TypeSignatureToken.Tuple);
                foreach (var field in t.Fields)
                {
                    Serialize(writer, field);
                }

                writer.Write((byte)TypeSignatureToken.End);
                break;
            default:
                throw new ArgumentException($"Unsupported type: {type}");
        }
    }

    public static void Serialize(BinaryWriter writer, Shape shape)
    {
        if (shape.IsScalar)
        {
            writer.Write((byte)0);
        }
        else
        {
            writer.Write((byte)1);
            foreach (var dim in shape)
            {
                writer.Write((byte)dim.Kind);
                if (dim.IsFixed)
                {
                    writer.Write(dim.FixedValue);
                }
            }

            writer.Write((byte)TypeSignatureToken.End);
        }
    }
}
