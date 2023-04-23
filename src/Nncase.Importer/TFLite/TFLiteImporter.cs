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

namespace Nncase.Importer.TFLite;

/// <summary>
/// TFLite importer.
/// </summary>
public sealed partial class TFLiteImporter : BaseImporter
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

    private readonly tflite.Model _model;
    private readonly tflite.SubGraph _subGraph;
    private readonly Dictionary<int, Expr> _outputTensors = new Dictionary<int, Expr>();

    /// <summary>
    /// Initializes a new instance of the <see cref="TFLiteImporter"/> class.
    /// </summary>
    /// <param name="tfliteModel">TFLite model bytes.</param>
    /// <param name="compileSession">Compile session.</param>
    public TFLiteImporter(byte[] tfliteModel, CompileSession compileSession)
        : base(compileSession)
    {
        _model = tflite.Model.GetRootAsModel(new ByteBuffer(tfliteModel));
        if (!tflite.Model.ModelBufferHasIdentifier(_model.ByteBuffer))
        {
            throw new InvalidDataException("Invalid tflite model file.");
        }

        _subGraph = _model.Subgraphs(0)!.Value;
    }

    /// <inheritdoc/>
    protected override (IEnumerable<Var> Inputs, Dictionary<Var, Expr[]> VarMap) CreateInputs()
    {
        var inputsCount = _subGraph.InputsLength;
        var created_inputs = new Var[inputsCount];
        for (int i = 0; i < inputsCount; i++)
        {
            var inputId = _subGraph.Inputs(i);
            var tensor = _subGraph.Tensors(inputId)!.Value;
            var input = new Var(tensor.Name, GetIRType(tensor));
            created_inputs[i] = input;
            _outputTensors.Add(inputId, input);
        }

        return (created_inputs, new());
    }

    protected override void ConvertOp()
    {
        for (int i = 0; i < _subGraph.OperatorsLength; i++)
        {
            var op = _subGraph.Operators(i)!.Value;
            Visit(op);
        }
    }

    protected override Expr CreateOutputs()
    {
        var outputs = (from o in _subGraph.GetOutputsBytes().AsValueEnumerable()
                       select _outputTensors[o]).ToArray();
        var outputTuple = new IR.Tuple(outputs);
        return outputTuple;
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
            tensor.Shape(i) == -1 ? Dimension.Unknown : tensor.Shape(i)).ToArray();
    }

    private void Visit(in tflite.Operator op)
    {
        // Compatible with older version model
        var opcode = _model.OperatorCodes((int)op.OpcodeIndex)!.Value;
        var builtinCode = (tflite.BuiltinOperator)Math.Max(
            opcode.DeprecatedBuiltinCode,
            (int)opcode.BuiltinCode);
        AddOpInModel(builtinCode.ToString());

        var output = builtinCode switch
        {
            tflite.BuiltinOperator.ABS => VisitUnary(op, UnaryOp.Abs),
            tflite.BuiltinOperator.ADD => VisitBinary(op, BinaryOp.Add, op.BuiltinOptionsAsAddOptions().FusedActivationFunction),

            // tflite.BuiltinOperator.ADD_N,
            tflite.BuiltinOperator.ARG_MAX => VisitReduceArg(op, ReduceArgOp.ArgMax),
            tflite.BuiltinOperator.ARG_MIN => VisitReduceArg(op, ReduceArgOp.ArgMin),

            // tflite.BuiltinOperator.ASSIGN_VARIABLE,
            tflite.BuiltinOperator.AVERAGE_POOL_2D => VisitReduceWindow2D(op, ReduceOp.Mean, 0f),

            tflite.BuiltinOperator.BATCH_MATMUL => VisitMatMul(op, false),
            tflite.BuiltinOperator.BATCH_TO_SPACE_ND => VisitBatchToSpaceND(op),

            // tflite.BuiltinOperator.BIDIRECTIONAL_SEQUENCE_LSTM,
            // tflite.BuiltinOperator.BIDIRECTIONAL_SEQUENCE_RNN,
            // tflite.BuiltinOperator.BROADCAST_TO,
            // tflite.BuiltinOperator.CALL,
            // tflite.BuiltinOperator.CALL_ONCE,
            tflite.BuiltinOperator.CAST => VisitCast(op),
            tflite.BuiltinOperator.CEIL => VisitUnary(op, UnaryOp.Ceil),

            // tflite.BuiltinOperator.COMPLEX_ABS,
            tflite.BuiltinOperator.CONCATENATION => VisitConcat(op),

            // tflite.BuiltinOperator.CONCAT_EMBEDDINGS,
            tflite.BuiltinOperator.CONV_2D => VisitConv2D(op),

            // tflite.BuiltinOperator.CONV_3D,
            // tflite.BuiltinOperator.CONV_3D_TRANSPOSE,
            tflite.BuiltinOperator.COS => VisitUnary(op, UnaryOp.Cos),

            // tflite.BuiltinOperator.CUMSUM,
            // tflite.BuiltinOperator.CUSTOM,
            // tflite.BuiltinOperator.DELEGATE,
            // tflite.BuiltinOperator.DENSIFY,
            tflite.BuiltinOperator.DEPTHWISE_CONV_2D => VisitDepthwiseConv2D(op),

            // tflite.BuiltinOperator.DEPTH_TO_SPACE,
            tflite.BuiltinOperator.DEQUANTIZE => VisitDeQuantize(op),
            tflite.BuiltinOperator.DIV => VisitBinary(op, BinaryOp.Div, op.BuiltinOptionsAsDivOptions().FusedActivationFunction),

            // tflite.BuiltinOperator.ELU,
            // tflite.BuiltinOperator.EMBEDDING_LOOKUP,
            // tflite.BuiltinOperator.EMBEDDING_LOOKUP_SPARSE,
            // tflite.BuiltinOperator.EQUAL,
            tflite.BuiltinOperator.EXP => VisitUnary(op, UnaryOp.Exp),

            tflite.BuiltinOperator.EXPAND_DIMS => VisitExpandDims(op),

            // tflite.BuiltinOperator.FAKE_QUANT,
            tflite.BuiltinOperator.FILL => VisitFill(op),
            tflite.BuiltinOperator.FLOOR => VisitUnary(op, UnaryOp.Ceil),

            tflite.BuiltinOperator.FLOOR_DIV => VisitFloorDiv(op),
            tflite.BuiltinOperator.FLOOR_MOD => VisitFloorMod(op),

            tflite.BuiltinOperator.FULLY_CONNECTED => VisitMatMul(op),
            tflite.BuiltinOperator.GATHER => VisitGather(op),
            tflite.BuiltinOperator.GATHER_ND => VisitGatherND(op),

            tflite.BuiltinOperator.GREATER => VisitCompare(op, CompareOp.GreaterThan),
            tflite.BuiltinOperator.GREATER_EQUAL => VisitCompare(op, CompareOp.GreaterOrEqual),

            tflite.BuiltinOperator.HARD_SWISH => VisitHardSwish(op),

            // tflite.BuiltinOperator.HASHTABLE,
            // tflite.BuiltinOperator.HASHTABLE_FIND,
            // tflite.BuiltinOperator.HASHTABLE_IMPORT,
            // tflite.BuiltinOperator.HASHTABLE_LOOKUP,
            // tflite.BuiltinOperator.HASHTABLE_SIZE,
            // tflite.BuiltinOperator.IF,
            // tflite.BuiltinOperator.IMAG,
            tflite.BuiltinOperator.L2_NORMALIZATION => VisitL2Normalization(op),

            // tflite.BuiltinOperator.L2_POOL_2D,
            tflite.BuiltinOperator.LEAKY_RELU => VisitLeakyRelu(op),

            tflite.BuiltinOperator.LESS => VisitCompare(op, CompareOp.LowerThan),

            // tflite.BuiltinOperator.LESS_EQUAL,
            // tflite.BuiltinOperator.LOCAL_RESPONSE_NORMALIZATION,
            tflite.BuiltinOperator.LOG => VisitUnary(op, UnaryOp.Log),

            // tflite.BuiltinOperator.LOGICAL_AND,
            // tflite.BuiltinOperator.LOGICAL_NOT,
            // tflite.BuiltinOperator.LOGICAL_OR,
            tflite.BuiltinOperator.LOGISTIC => VisitLogistic(op),

            tflite.BuiltinOperator.LOG_SOFTMAX => VisitLogSoftMax(op),

            // tflite.BuiltinOperator.LSH_PROJECTION,
            // tflite.BuiltinOperator.LSTM,
            // tflite.BuiltinOperator.MATRIX_DIAG,
            // tflite.BuiltinOperator.MATRIX_SET_DIAG,
            tflite.BuiltinOperator.MAXIMUM => VisitBinary(op, BinaryOp.Max),

            tflite.BuiltinOperator.MAX_POOL_2D => VisitReduceWindow2D(op, ReduceOp.Max, float.MinValue),
            tflite.BuiltinOperator.MEAN => VisitReduce(op, ReduceOp.Mean, 0f),
            tflite.BuiltinOperator.MINIMUM => VisitBinary(op, BinaryOp.Min),

            tflite.BuiltinOperator.MIRROR_PAD => VisitMirrorPad(op),
            tflite.BuiltinOperator.MUL => VisitBinary(op, BinaryOp.Mul, op.BuiltinOptionsAsMulOptions().FusedActivationFunction),
            tflite.BuiltinOperator.NEG => VisitUnary(op, UnaryOp.Neg),

            // tflite.BuiltinOperator.NON_MAX_SUPPRESSION_V4,
            // tflite.BuiltinOperator.NON_MAX_SUPPRESSION_V5,
            tflite.BuiltinOperator.NOT_EQUAL => VisitNotEqual(op),
            tflite.BuiltinOperator.ONE_HOT => VisitOneHot(op),
            tflite.BuiltinOperator.PACK => VisitPack(op),
            tflite.BuiltinOperator.PAD => VisitPad(op),
            tflite.BuiltinOperator.PADV2 => VisitPadV2(op),

            // tflite.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES,
            tflite.BuiltinOperator.POW => VisitBinary(op, BinaryOp.Pow),

            tflite.BuiltinOperator.PRELU => VisitPRelu(op),
            tflite.BuiltinOperator.QUANTIZE => VisitQuantize(op),

            tflite.BuiltinOperator.RANGE => VisitRange(op),

            // tflite.BuiltinOperator.RANK,
            // tflite.BuiltinOperator.READ_VARIABLE,
            // tflite.BuiltinOperator.REAL,
            // tflite.BuiltinOperator.REDUCE_ALL,
            // tflite.BuiltinOperator.REDUCE_ANY,
            tflite.BuiltinOperator.REDUCE_MAX => VisitReduce(op, ReduceOp.Max, float.MinValue),
            tflite.BuiltinOperator.REDUCE_MIN => VisitReduce(op, ReduceOp.Min, float.MaxValue),

            tflite.BuiltinOperator.REDUCE_PROD => VisitReduce(op, ReduceOp.Prod, 1f),
            tflite.BuiltinOperator.RELU => VisitRelu(op),
            tflite.BuiltinOperator.RELU6 => VisitRelu6(op),

            // tflite.BuiltinOperator.RELU_N1_TO_1,
            tflite.BuiltinOperator.RESHAPE => VisitReshape(op),
            tflite.BuiltinOperator.RESIZE_BILINEAR => VisitResizeImage(op, ImageResizeMode.Bilinear),
            tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR => VisitResizeImage(op, ImageResizeMode.NearestNeighbor),

            // tflite.BuiltinOperator.REVERSE_SEQUENCE,
            // tflite.BuiltinOperator.REVERSE_V2,
            // tflite.BuiltinOperator.RFFT2D,
            // tflite.BuiltinOperator.RNN,
            tflite.BuiltinOperator.ROUND => VisitUnary(op, UnaryOp.Round),
            tflite.BuiltinOperator.RSQRT => VisitUnary(op, UnaryOp.Rsqrt),

            // tflite.BuiltinOperator.SCATTER_ND,
            // tflite.BuiltinOperator.SEGMENT_SUM,
            // tflite.BuiltinOperator.SELECT,
            // tflite.BuiltinOperator.SELECT_V2,
            tflite.BuiltinOperator.SHAPE => VisitShape(op),
            tflite.BuiltinOperator.SIN => VisitUnary(op, UnaryOp.Sin),

            // tflite.BuiltinOperator.SKIP_GRAM,
            tflite.BuiltinOperator.SLICE => VisitSlice(op),
            tflite.BuiltinOperator.SOFTMAX => VisitSoftMax(op),
            tflite.BuiltinOperator.SPACE_TO_BATCH_ND => VisitSpaceToBatchND(op),

            // tflite.BuiltinOperator.SPACE_TO_DEPTH,
            // tflite.BuiltinOperator.SPARSE_TO_DENSE,
            tflite.BuiltinOperator.SPLIT => VisitSplit(op),

            // tflite.BuiltinOperator.SPLIT_V,
            tflite.BuiltinOperator.SQRT => VisitUnary(op, UnaryOp.Sqrt),
            tflite.BuiltinOperator.SQUARE => VisitUnary(op, UnaryOp.Square),

            tflite.BuiltinOperator.SQUARED_DIFFERENCE => VisitSquareDifference(op),
            tflite.BuiltinOperator.SQUEEZE => VisitSqueeze(op),
            tflite.BuiltinOperator.STRIDED_SLICE => VisitStrideSlice(op),
            tflite.BuiltinOperator.SUB => VisitBinary(op, BinaryOp.Sub, op.BuiltinOptionsAsSubOptions().FusedActivationFunction),

            tflite.BuiltinOperator.SUM => VisitReduce(op, ReduceOp.Sum, 0f),

            // tflite.BuiltinOperator.SVDF,
            tflite.BuiltinOperator.TANH => VisitUnary(op, UnaryOp.Tanh),

            tflite.BuiltinOperator.TILE => VisitTile(op),

            // tflite.BuiltinOperator.TOPK_V2,
            tflite.BuiltinOperator.TRANSPOSE => VisitTranspose(op),
            tflite.BuiltinOperator.TRANSPOSE_CONV => VisitConv2DTranspose(op),

            // tflite.BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM,
            // tflite.BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_RNN,
            // tflite.BuiltinOperator.UNIQUE,
            // tflite.BuiltinOperator.UNPACK,
            // tflite.BuiltinOperator.VAR_HANDLE,
            tflite.BuiltinOperator.WHERE => VisitWhere(op),

            // tflite.BuiltinOperator.WHILE,
            // tflite.BuiltinOperator.ZEROS_LIKE,
            _ => UnSupportedOp(builtinCode.ToString()),
        };

        List<string> outputNames = new();

        var outputsLength = op.GetOutputsArray().Length;
        for (int i = 0; i < outputsLength; i++)
        {
            outputNames.Add(GetOutputTensor(op, i).Name);
        }

        output.Metadata.OutputNames = outputNames;

        AddToOutputs(_outputTensors, op.GetOutputsArray(), output);
    }

    private List<QuantParam>? GetInputQuantParams(in tflite.Operator op, int index)
    {
        var id = op.Inputs(index);
        var quantParams = new List<QuantParam>();

        if (id > _subGraph.TensorsLength)
        {
            throw new InvalidDataException($"Cannot find tensor (id:{id}).");
        }

        // Maybe constant
        var tensor = _subGraph.Tensors(id) ?? throw new InvalidDataException($"Cannot find tensor (id:{id}).");
        if (!tensor.Quantization.HasValue
            || tensor.Quantization.Value.QuantizedDimension == 0)
        {
            return null;
        }
        else
        {
            var quantParam = tensor.Quantization.Value;

            // Only support by tensor quant now.
            Trace.Assert(quantParam.ZeroPointLength == 1);
            for (var i = 0; i < quantParam.ZeroPointLength; i++)
            {
                quantParams.Add(new QuantParam((int)quantParam.GetZeroPointArray()[i], quantParam.GetScaleArray()[i]));
            }

            return quantParams;
        }
    }

    private List<QuantParam>? GetOutputQuantParams(in tflite.Operator op, int index)
    {
        var id = op.Outputs(index);
        var quantParams = new List<QuantParam>();

        if (id > _subGraph.TensorsLength)
        {
            throw new InvalidDataException($"Cannot find tensor (id:{id}).");
        }

        var tensor = _subGraph.Tensors(id) ?? throw new InvalidDataException($"Cannot find tensor (id:{id}).");
        if (!tensor.Quantization.HasValue
            || tensor.Quantization.Value.QuantizedDimension == 0)
        {
            return null;
        }
        else
        {
            var quantParam = tensor.Quantization.Value;

            // Only support by tensor quant now.
            System.Diagnostics.Trace.Assert(quantParam.ZeroPointLength == 1);
            for (var i = 0; i < quantParam.ZeroPointLength; i++)
            {
                quantParams.Add(new QuantParam((int)quantParam.GetZeroPointArray()[i], quantParam.GetScaleArray()[i]));
            }

            return quantParams;
        }
    }

    private Expr GetInputExprs(in tflite.Operator op, int index)
    {
        var id = op.Inputs(index);

        if (_outputTensors.TryGetValue(id, out var expr))
        {
            expr.Metadata.OutputNames = new string[] { GetInputTensor(op, index).Name };
            return expr;
        }
        else
        {
            if (id > _subGraph.TensorsLength)
            {
                throw new InvalidDataException($"Cannot find tensor (id:{id}).");
            }

            // Maybe constant
            var tensor = _subGraph.Tensors(id) ?? throw new InvalidDataException($"Cannot find tensor (id:{id}).");
            var buffer = _model.Buffers((int)tensor.Buffer) ?? throw new InvalidDataException($"Cannot find buffer (id:{tensor.Buffer}).");
            var data = buffer.GetDataBytes();
            if (!data.IsEmpty)
            {
                var con = Const.FromTensor(Tensor.FromBytes(GetIRType(tensor), data.ToArray()));
                con.Metadata.OutputNames = new string[] { GetInputTensor(op, index).Name };
                _outputTensors.Add(id, con);
                return con;
            }
            else
            {
                throw new InvalidDataException($"Cannot load tensor data (tensor:{tensor.Name}).");
            }
        }
    }

    private (Expr Expr0, Expr Expr1) GetInputExprs(in tflite.Operator op, int index0, int index1) =>
        (GetInputExprs(op, index0), GetInputExprs(op, index1));

    private tflite.Tensor GetTfliteTensor(int id)
    {
        var output = _subGraph.Tensors(id) ??
                     throw new InvalidDataException($"Cannot find tensor (id:{id}).");
        return output;
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
