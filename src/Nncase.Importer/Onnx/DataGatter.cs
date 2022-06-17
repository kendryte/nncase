using System;
using System.Collections.Generic;
using System.IO;
using Nncase.IR;
using System.Linq;
using LanguageExt;
using Onnx;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer;

public sealed partial class OnnxImporter
{
    protected static readonly Dictionary<TensorProto.Types.DataType, DataType> _typeMap = new()
    {
        {TensorProto.Types.DataType.Bool, DataTypes.Boolean},
        {TensorProto.Types.DataType.Float16, DataTypes.Float16},
        {TensorProto.Types.DataType.Float, DataTypes.Float32},
        {TensorProto.Types.DataType.Double, DataTypes.Float64},
        {TensorProto.Types.DataType.Int16, DataTypes.Int16},
        {TensorProto.Types.DataType.Int32, DataTypes.Int32},
        {TensorProto.Types.DataType.Int64, DataTypes.Int64},
        {TensorProto.Types.DataType.Int8, DataTypes.Int8},
        {TensorProto.Types.DataType.String, DataTypes.Utf8Char},
        {TensorProto.Types.DataType.Uint32, DataTypes.UInt32},
        {TensorProto.Types.DataType.Uint64, DataTypes.UInt64},
        {TensorProto.Types.DataType.Uint8, DataTypes.UInt8},
    };

    protected bool EmptyTensor(TensorProto tensor)
    {
        return tensor.Dims.Count == 1 && tensor.Dims[0] == 0;
    }

    protected Tensor GetTensor(TensorProto tensor)
    {
        var shape = GetShape(tensor).ToValueArray();
        var type = GetDataType(tensor);
        var dt = (TensorProto.Types.DataType) tensor.DataType;

        // should not use tensor.DataLocation to distinguish whether it is RawData
        if (tensor.RawData.ToByteArray().Length() != 0)
        {
            return Tensor.FromBytes(type, tensor.RawData.ToByteArray(), shape);
        }
        else
        {
            return dt switch
            {
                // todo:not directly supported type should convert
                //TensorProto.Types.DataType.Bool => Tensor.FromSpan(),
                //TensorProto.Types.DataType.Float16 => Tensor.FromSpan(),
                TensorProto.Types.DataType.Float => Tensor.FromSpan<float>(tensor.FloatData.ToArray(), shape),
                TensorProto.Types.DataType.Double => Tensor.FromSpan<double>(tensor.DoubleData.ToArray(), shape),

                //TensorProto.Types.DataType.Int16 => Tensor.FromSpan(),
                TensorProto.Types.DataType.Int32 => Tensor.FromSpan<int>(tensor.Int32Data.ToArray(), shape),
                TensorProto.Types.DataType.Int64 => Tensor.FromSpan<long>(tensor.Int64Data.ToArray(), shape),

                //TensorProto.Types.DataType.Int8 => Tensor.FromSpan(),
                //TensorProto.Types.DataType.String => Tensor.FromSpan(),
                //TensorProto.Types.DataType.Uint32 => Tensor.FromSpan(),
                //TensorProto.Types.DataType.Uint64 => Tensor.FromSpan<ulong>(tensor.Uint64Data.ToArray(), shape),
                //TensorProto.Types.DataType.Uint8 => Tensor.FromSpan(),
                _ => throw new NotSupportedException($"Not supported onnx constant data type{dt}"),
            };
        }
    }

    public Shape GetShape(ValueInfoProto v)
    {
        var shape = v.Type.TensorType.Shape.Dim.Select(x => x.DimValue);
        return new Shape(shape);
    }

    public Shape GetShape(TensorProto tensor)
    {
        return new Shape(tensor.Dims);
    }

    public TensorType GetIRType(ValueInfoProto v)
    {
        return new TensorType(GetDataType(v), GetShape(v));
    }

    public TensorType GetIRType(TensorProto v)
    {
        return new TensorType(GetDataType(v), GetShape(v));
    }

    protected Expr GetInputExpr(NodeProto n, int index)
    {
        // todo:is null?
        var id = n.Input[index];
        if (_outputTensors.TryGetValue(id, out var expr))
        {
            return expr;
        }

        return _graph.Initializer
            .Find(x => x.Name == id)
            .Match(
                GetTensor,
                () => throw new InvalidDataException($"Cannot load tensor data (tensor:{id})."));
    }

    protected DataType GetInputDataType(NodeProto n, int index)
    {
        var id = n.Input[index];
        return _graph.Input.Concat(_graph.ValueInfo)
            .Find(x => x.Name == id)
            .Match(GetDataType,
                () => throw new InvalidDataException($"Cannot load tensor data (tensor:{id})."));
    }

    protected Expr GetSingleInputExpr(NodeProto n)
    {
        return GetInputExpr(n, 0);
    }

    protected (Expr, Expr) GetInputExprs(NodeProto n, int index0, int index1)
    {
        return (GetInputExpr(n, index0), GetInputExpr(n, index1));
    }

    protected Option<Expr> GetOptionInputExpr(NodeProto n, int index)
    {
        if (n.Input.Count <= index)
        {
            return Option<Expr>.None;
        }

        var id = n.Input[index];
        if (id == "")
        {
            return Option<Expr>.None;
        }

        if (_outputTensors.TryGetValue(id, out var expr))
        {
            return expr;
        }

        return _graph.Initializer
            .Find(x => x.Name == id)
            .Match(
                t => EmptyTensor(t) ? Option<Expr>.None : Option<Expr>.Some(GetTensor(t)),
                () => throw new InvalidDataException($"Cannot load tensor data (tensor:{id})."));
    }

    protected Expr GetOptionInputExpr(NodeProto n, int index, Expr defaultExpr)
    {
        return GetOptionInputExpr(n, index).Or(defaultExpr);
    }

    protected (Option<Expr>, Option<Expr>) GetOptionInputExprs(NodeProto n, int index0, int index1)
    {
        return (GetOptionInputExpr(n, index0), GetOptionInputExpr(n, index1));
    }

    protected Expr ToNncasePadFormat(Expr pads)
    {
        return Transpose(Reshape(pads, new[] {-1, 2}), new[] {1, 0});
    }
}
