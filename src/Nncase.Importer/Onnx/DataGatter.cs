// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Google.Protobuf.Collections;
using LanguageExt;
using Nncase.IR;
using Onnx;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer;

public sealed partial class OnnxImporter
{
    private static readonly Dictionary<TensorProto.Types.DataType, DataType> _typeMap = new()
    {
        { TensorProto.Types.DataType.Bool, DataTypes.Boolean },
        { TensorProto.Types.DataType.Float16, DataTypes.Float16 },
        { TensorProto.Types.DataType.Float, DataTypes.Float32 },
        { TensorProto.Types.DataType.Double, DataTypes.Float64 },
        { TensorProto.Types.DataType.Int16, DataTypes.Int16 },
        { TensorProto.Types.DataType.Int32, DataTypes.Int32 },
        { TensorProto.Types.DataType.Int64, DataTypes.Int64 },
        { TensorProto.Types.DataType.Int8, DataTypes.Int8 },
        { TensorProto.Types.DataType.String, DataTypes.Utf8Char },
        { TensorProto.Types.DataType.Uint32, DataTypes.UInt32 },
        { TensorProto.Types.DataType.Uint64, DataTypes.UInt64 },
        { TensorProto.Types.DataType.Uint8, DataTypes.UInt8 },
    };

    public Shape GetShape(ValueInfoProto v)
    {
        var shape = v.Type.TensorType.Shape.Dim;
        var dimArr = GetDimArray(shape, d => d, _ => Dimension.Unknown, d => (Dimension)d.DimValue);
        return new Shape(dimArr);
    }

    public Expr[] GetOriginShape(ValueInfoProto v)
    {
        var shape = v.Type.TensorType.Shape.Dim;
        return GetDimArray(shape, d => d, dim => _dynVarMap[dim.DimParam], dim => (Expr)dim.DimValue);
    }

    public T[] GetDimArray<T>(
        RepeatedField<TensorShapeProto.Types.Dimension> shape,
        Func<int, T> fixVarF,
        Func<TensorShapeProto.Types.Dimension, T> dynamicF,
        Func<TensorShapeProto.Types.Dimension, T> fixF)
    {
        return shape.Select(x =>
        {
            if (IsDynamicDim(x))
            {
                if (_fixVarMap.TryGetValue(x.DimParam, out var dim))
                {
                    return fixVarF(dim);
                }

                return dynamicF(x);
            }

            return fixF(x);
        }).ToArray();
    }

    public Shape GetShape(TensorProto tensor)
    {
        return new Shape(tensor.Dims.ToArray());
    }

    public TensorType GetIRType(ValueInfoProto v)
    {
        return new TensorType(GetDataType(v), GetShape(v));
    }

    public TensorType GetIRType(TensorProto v)
    {
        return new TensorType(GetDataType(v), GetShape(v));
    }

    private static bool IsDynamicDim(TensorShapeProto.Types.Dimension x) => x.DimParam != string.Empty;

    private bool EmptyTensor(TensorProto tensor)
    {
        return tensor.Dims.Count == 1 && tensor.Dims[0] == 0;
    }

    private Tensor GetExternalTensor<T>(BinaryReader br, DataType dataType, long length, Shape shape)
        where T : unmanaged, IEquatable<T>
    {
        var tensorArray = new T[length / dataType.SizeInBytes];
        var totalRead = 0;
        int chunk = 1024 * 1024 * 1024;
        for (long l = length; l > 0; l -= chunk)
        {
            var tmpBuffer = br.ReadBytes((int)Math.Min(chunk, l));

            Buffer.BlockCopy(tmpBuffer, 0, tensorArray, totalRead, tmpBuffer.Length);
            totalRead += tmpBuffer.Length / dataType.SizeInBytes;
        }

        return Tensor.From(tensorArray, shape);
    }

    private Tensor GetTensor(TensorProto tensor)
    {
        var shape = GetShape(tensor).ToValueArray();
        var type = GetDataType(tensor);
        var dt = (TensorProto.Types.DataType)tensor.DataType;

        // should not use tensor.DataLocation to distinguish whether it is RawData
        if (tensor.RawData.ToByteArray().Length() != 0)
        {
            return Tensor.FromBytes(type, tensor.RawData.ToByteArray(), shape);
        }

        // model size > 2G
        // https://github.com/onnx/onnx/blob/main/docs/ExternalData.md
        var externalDataCount = tensor.ExternalData.Count;
        if (externalDataCount != 0)
        {
            if (externalDataCount < 1 || externalDataCount > 5)
            {
                throw new NotSupportedException("NotSupport ExternalData format, only support location, offset, length, checksum");
            }

            var parent = Directory.GetParent(CompileSession.CompileOptions.InputFile)?.FullName;
            var externalData = tensor.ExternalData;
            var location = Path.Join(parent, externalData[0].Value);
            var offset = externalDataCount > 1L ? long.Parse(externalData[1].Value) : 0;
            using var br = new BinaryReader(new FileStream(location, FileMode.Open));
            var length = externalDataCount > 1 ? long.Parse(externalData[2].Value) : br.BaseStream.Length;
            br.BaseStream.Seek(offset, SeekOrigin.Begin);

            return type switch
            {
                var t when t == DataTypes.Float32 => GetExternalTensor<float>(br, type, length, shape),
                var t when t == DataTypes.Float64 => GetExternalTensor<double>(br, type, length, shape),
                var t when t == DataTypes.Int32 => GetExternalTensor<int>(br, type, length, shape),
                var t when t == DataTypes.Int64 => GetExternalTensor<long>(br, type, length, shape),
                var t when t == DataTypes.Int8 => GetExternalTensor<sbyte>(br, type, length, shape),
                var t when t == DataTypes.UInt8 => GetExternalTensor<byte>(br, type, length, shape),
                _ => throw new NotSupportedException($"Not supported onnx constant data type {type}"),
            };
        }

        return dt switch
        {
            // todo:not directly supported type should convert
            // TensorProto.Types.DataType.Bool => Tensor.FromSpan(),
            // TensorProto.Types.DataType.Float16 => Tensor.FromSpan(),
            TensorProto.Types.DataType.Float => Tensor.From<float>(tensor.FloatData.ToArray(), shape),
            TensorProto.Types.DataType.Double => Tensor.From<double>(tensor.DoubleData.ToArray(), shape),

            // TensorProto.Types.DataType.Int16 => Tensor.FromSpan(),
            TensorProto.Types.DataType.Int32 => Tensor.From<int>(tensor.Int32Data.ToArray(), shape),
            TensorProto.Types.DataType.Int64 => Tensor.From<long>(tensor.Int64Data.ToArray(), shape),

            TensorProto.Types.DataType.Int8 => Tensor.From<sbyte>(
                tensor.Int32Data.Select(x => (sbyte)x).ToArray(),
                shape),

            // TensorProto.Types.DataType.String => Tensor.FromSpan(),
            // TensorProto.Types.DataType.Uint32 => Tensor.FromSpan(),
            // TensorProto.Types.DataType.Uint64 => Tensor.FromSpan<ulong>(tensor.Uint64Data.ToArray(), shape),
            TensorProto.Types.DataType.Uint8 => Tensor.From<byte>(
                tensor.Int32Data.Select(x => (byte)x).ToArray(),
                shape),
            _ => throw new NotSupportedException($"Not supported onnx constant data type{dt}"),
        };
    }

    private Expr GetInputExpr(NodeProto n, int index)
    {
        // todo:is null?
        var id = n.Input[index];
        if (_outputTensors!.TryGetValue(id, out var expr))
        {
            expr.Metadata.OutputNames = new string[] { n.Input[index] };
            return expr;
        }

        Expr ret = _graph.Initializer
            .Find(x => x.Name == id)
            .Match(
                GetTensor,
                () => throw new InvalidDataException($"Cannot load tensor data (tensor:{id})."));
        ret.Metadata.OutputNames = new string[] { n.Input[index] };
        return ret;
    }

    private DataType GetInputDataType(NodeProto n, int index)
    {
        var id = n.Input[index];
        return _graph.Input.Concat(_graph.ValueInfo)
            .Find(x => x.Name == id)
            .Match(
                GetDataType,
                () => throw new InvalidDataException($"Cannot load tensor data (tensor:{id})."));
    }

    private Expr GetSingleInputExpr(NodeProto n)
    {
        return GetInputExpr(n, 0);
    }

    private DataType GetOutputType(NodeProto n)
    {
        var outName = n.Output[0];
        return _graph.Output.Concat(_graph.ValueInfo).Find(node => node.Name == outName)
            .Match(
                n => GetDataType(n),
                () => throw new InvalidOperationException($"Can't find Output for node:{n.Name}"));
    }

    private (Expr Input1, Expr Input2) GetInputExprs(NodeProto n, int index0, int index1)
    {
        return (GetInputExpr(n, index0), GetInputExpr(n, index1));
    }

    private Option<Expr> GetOptionInputExpr(NodeProto n, int index)
    {
        if (n.Input.Count <= index)
        {
            return Option<Expr>.None;
        }

        var id = n.Input[index];
        if (id == string.Empty)
        {
            return Option<Expr>.None;
        }

        if (_outputTensors!.TryGetValue(id, out var expr))
        {
            return expr;
        }

        return _graph.Initializer
            .Find(x => x.Name == id)
            .Match(
                t => EmptyTensor(t) ? Option<Expr>.None : Option<Expr>.Some(GetTensor(t)),
                () => throw new InvalidDataException($"Cannot load tensor data (tensor:{id})."));
    }

    private Expr GetOptionInputExpr(NodeProto n, int index, Expr defaultExpr)
    {
        return GetOptionInputExpr(n, index).Or(defaultExpr);
    }

    private (Option<Expr> Input1, Option<Expr> Input2) GetOptionInputExprs(NodeProto n, int index0, int index1)
    {
        return (GetOptionInputExpr(n, index0), GetOptionInputExpr(n, index1));
    }

    private Expr ToNncasePadFormat(Expr pads)
    {
        return Transpose(Reshape(pads, new[] { 2, -1 }), new[] { 1, 0 });
    }
}
