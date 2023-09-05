// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FlatBuffers;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Math = System.Math;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Importer.Ncnn;

/// <summary>
/// Ncnn importer.
/// </summary>
public sealed partial class NcnnImporter : BaseImporter
{
    private static readonly Dictionary<tflite.TensorType, DataType> _typeMap = new()
    {
        { tflite.TensorType.BOOL, DataTypes.Boolean },
        { tflite.TensorType.FLOAT16, DataTypes.Float16 },
        { tflite.TensorType.FLOAT32, DataTypes.Float32 },
        { tflite.TensorType.FLOAT64, DataTypes.Float64 },
        { tflite.TensorType.INT16, DataTypes.Int16 },
        { tflite.TensorType.INT32, DataTypes.Int32 },
        { tflite.TensorType.INT64, DataTypes.Int64 },
        { tflite.TensorType.INT8, DataTypes.Int8 },
        { tflite.TensorType.STRING, DataTypes.Utf8Char },
        { tflite.TensorType.UINT32, DataTypes.UInt32 },
        { tflite.TensorType.UINT64, DataTypes.UInt64 },
        { tflite.TensorType.UINT8, DataTypes.UInt8 },
    };

    private readonly NcnnModel _model;
    private readonly Dictionary<int, Expr> _outputTensors = new Dictionary<int, Expr>();

    /// <summary>
    /// Initializes a new instance of the <see cref="NcnnImporter"/> class.
    /// </summary>
    /// <param name="ncnnParam">Ncnn param stream.</param>
    /// <param name="ncnnBin">Ncnn bin stream.</param>
    /// <param name="compileSession">Compile session.</param>
    public NcnnImporter(Stream ncnnParam, Stream ncnnBin, CompileSession compileSession)
        : base(compileSession)
    {
        _model = NcnnModel.ParseFromStream(ncnnParam);
    }

    /// <inheritdoc/>
    protected override (IEnumerable<Var> Inputs, Dictionary<Var, Expr[]> VarMap) CreateInputs()
    {
        throw new NotImplementedException();
    }

    protected override void ConvertOp()
    {
        throw new NotImplementedException();
    }

    protected override Expr CreateOutputs()
    {
        throw new NotImplementedException();
    }

    /// <summary>
    /// Create IR type from tflite shape and tensor type.
    /// </summary>
    /// <param name="tensor">Tensor.</param>
    /// <returns>Created IR type.</returns>
    private static TensorType GetIRType(tflite.Tensor tensor)
    {
        var shape = GetShapeArray(tensor);
        var dataType = GetDataType(tensor.Type);
        if (shape.Length == 0)
        {
            return TensorType.Scalar(dataType);
        }
        else
        {
            return new TensorType(dataType, new Shape(shape));
        }
    }

    private static DataType GetDataType(tflite.TensorType type)
    {
        if (_typeMap.TryGetValue(type, out var dataType))
        {
            return dataType;
        }

        throw new NotSupportedException($"Unsupported tflite tensor type: {type}.");
    }

    private static Dimension[] GetShapeArray(tflite.Tensor tensor)
    {
        if (tensor.ShapeSignatureLength == 0)
        {
            return tensor.GetShapeArray().Select(x => new Dimension(x)).ToArray();
        }

        return Enumerable.Range(0, tensor.ShapeLength).Select(i =>
            tensor.ShapeSignature(i) < 0 ? Dimension.Unknown : tensor.Shape(i)).ToArray();
    }

    private void Visit(in tflite.Operator op)
    {
        throw new NotImplementedException();
    }

    private List<QuantParam>? GetInputQuantParams(in tflite.Operator op, int index)
    {
        throw new NotImplementedException();
    }

    private List<QuantParam>? GetOutputQuantParams(in tflite.Operator op, int index)
    {
        throw new NotImplementedException();
    }

    private Expr GetInputExprs(in tflite.Operator op, int index)
    {
        throw new NotImplementedException();
    }

    private (Expr Expr0, Expr Expr1) GetInputExprs(in tflite.Operator op, int index0, int index1) =>
        (GetInputExprs(op, index0), GetInputExprs(op, index1));

    private tflite.Tensor GetTfliteTensor(int id)
    {
        throw new NotImplementedException();
    }

    private tflite.Tensor GetInputTensor(in tflite.Operator op, int index)
    {
        return GetTfliteTensor(op.Inputs(index));
    }

    private tflite.Tensor GetOutputTensor(in tflite.Operator op, int index)
    {
        return GetTfliteTensor(op.Outputs(index));
    }

    private Shape GetTensorShape(in tflite.Tensor tensor)
    {
        return GetShapeArray(tensor);
    }
}
