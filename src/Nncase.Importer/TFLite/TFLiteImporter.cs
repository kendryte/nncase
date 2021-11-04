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

namespace Nncase.Importer.TFLite
{
    /// <summary>
    /// TFLite importer.
    /// </summary>
    public sealed partial class TFLiteImporter
    {
        private static readonly Dictionary<tflite.TensorType, DataType> _typeMap = new()
        {
            { tflite.TensorType.BOOL, DataType.Bool },
            { tflite.TensorType.FLOAT16, DataType.Float16 },
            { tflite.TensorType.FLOAT32, DataType.Float32 },
            { tflite.TensorType.FLOAT64, DataType.Float64 },
            { tflite.TensorType.INT16, DataType.Int16 },
            { tflite.TensorType.INT32, DataType.Int32 },
            { tflite.TensorType.INT64, DataType.Int64 },
            { tflite.TensorType.INT8, DataType.Int8 },
            { tflite.TensorType.STRING, DataType.String },
            { tflite.TensorType.UINT32, DataType.UInt32 },
            { tflite.TensorType.UINT64, DataType.UInt64 },
            { tflite.TensorType.UINT8, DataType.UInt8 },
        };

        private readonly tflite.Model _model;
        private readonly tflite.SubGraph _subGraph;
        private readonly Dictionary<int, Expr> _outputTensors = new Dictionary<int, Expr>();

        /// <summary>
        /// Initializes a new instance of the <see cref="TFLiteImporter"/> class.
        /// </summary>
        /// <param name="tfliteModel">TFLite model bytes.</param>
        public TFLiteImporter(byte[] tfliteModel)
        {
            _model = tflite.Model.GetRootAsModel(new ByteBuffer(tfliteModel));
            if (!tflite.Model.ModelBufferHasIdentifier(_model.ByteBuffer))
            {
                throw new InvalidDataException("Invalid tflite model file.");
            }

            _subGraph = _model.Subgraphs(0)!.Value;
        }

        /// <summary>
        /// Import an IR module from tflite model.
        /// </summary>
        /// <returns>Imported IR module.</returns>
        public Module Import()
        {
            // 1. Create inputs
            var inputsCount = _subGraph.InputsLength;
            var created_inputs = new Expr[inputsCount];
            for (int i = 0; i < inputsCount; i++)
            {
                var inputId = _subGraph.Inputs(i);
                var tensor = _subGraph.Tensors(inputId)!.Value;
                var input = new Var(tensor.Name, GetIRType(tensor.GetShapeBytes(), tensor.Type));
                created_inputs[i] = input;
                _outputTensors.Add(inputId, input);
            }

            // 2. Convert ops
            for (int i = 0; i < _subGraph.OperatorsLength; i++)
            {
                var op = _subGraph.Operators(i)!.Value;
                Visit(op);
            }

            // 3. Create outputs
            var outputs = (from o in _subGraph.GetOutputsBytes().AsValueEnumerable()
                           select _outputTensors[o]).ToArray();
            var outputTuple = new IR.Tuple(ImmutableArray.Create(outputs));
            var mainFunc = new Function("main", outputTuple, created_inputs);

            var module = new Module();
            module.Add(mainFunc);
            module.Entry = mainFunc;
            return module;
        }

        /// <summary>
        /// Create IR type from tflite shape and tensor type.
        /// </summary>
        /// <param name="shape">Shape.</param>
        /// <param name="type">Tensor type.</param>
        /// <returns>Created IR type.</returns>
        private static TensorType GetIRType(Span<int> shape, tflite.TensorType type)
        {
            var dataType = GetDataType(type);
            if (shape.IsEmpty)
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

        private void Visit(in tflite.Operator op)
        {
            // Compatible with older version model
            var opcode = _model.OperatorCodes((int)op.OpcodeIndex)!.Value;
            var builtinCode = (tflite.BuiltinOperator)Math.Max(
                opcode.DeprecatedBuiltinCode,
                (int)opcode.BuiltinCode);

            var output = builtinCode switch
            {
                tflite.BuiltinOperator.ABS => VisitUnary(op, UnaryOp.Abs),
                tflite.BuiltinOperator.ADD => VisitBinary(op, BinaryOp.Add, op.BuiltinOptionsAsAddOptions().FusedActivationFunction),

                // tflite.BuiltinOperator.ADD_N,
                // tflite.BuiltinOperator.ARG_MAX,
                // tflite.BuiltinOperator.ARG_MIN,
                // tflite.BuiltinOperator.ASSIGN_VARIABLE,
                // tflite.BuiltinOperator.AVERAGE_POOL_2D,
                // tflite.BuiltinOperator.BATCH_MATMUL,
                // tflite.BuiltinOperator.BATCH_TO_SPACE_ND,
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
                // tflite.BuiltinOperator.DEPTHWISE_CONV_2D,
                // tflite.BuiltinOperator.DEPTH_TO_SPACE,
                // tflite.BuiltinOperator.DEQUANTIZE,
                tflite.BuiltinOperator.DIV => VisitBinary(op, BinaryOp.Div, op.BuiltinOptionsAsDivOptions().FusedActivationFunction),

                // tflite.BuiltinOperator.ELU,
                // tflite.BuiltinOperator.EMBEDDING_LOOKUP,
                // tflite.BuiltinOperator.EMBEDDING_LOOKUP_SPARSE,
                // tflite.BuiltinOperator.EQUAL,
                tflite.BuiltinOperator.EXP => VisitUnary(op, UnaryOp.Exp),

                // tflite.BuiltinOperator.EXPAND_DIMS,
                // tflite.BuiltinOperator.FAKE_QUANT,
                // tflite.BuiltinOperator.FILL,
                tflite.BuiltinOperator.FLOOR => VisitUnary(op, UnaryOp.Ceil),

                tflite.BuiltinOperator.FLOOR_DIV => VisitFloorDiv(op),
                tflite.BuiltinOperator.FLOOR_MOD => VisitFloorMod(op),

                tflite.BuiltinOperator.FULLY_CONNECTED => VisitMatMul(op),
                tflite.BuiltinOperator.GATHER => VisitGather(op),
                tflite.BuiltinOperator.GATHER_ND => VisitGatherND(op),
                // tflite.BuiltinOperator.GREATER,
                // tflite.BuiltinOperator.GREATER_EQUAL,
                // tflite.BuiltinOperator.HARD_SWISH,
                // tflite.BuiltinOperator.HASHTABLE,
                // tflite.BuiltinOperator.HASHTABLE_FIND,
                // tflite.BuiltinOperator.HASHTABLE_IMPORT,
                // tflite.BuiltinOperator.HASHTABLE_LOOKUP,
                // tflite.BuiltinOperator.HASHTABLE_SIZE,
                // tflite.BuiltinOperator.IF,
                // tflite.BuiltinOperator.IMAG,
                // tflite.BuiltinOperator.L2_NORMALIZATION,
                // tflite.BuiltinOperator.L2_POOL_2D,
                // tflite.BuiltinOperator.LEAKY_RELU,
                // tflite.BuiltinOperator.LESS,
                // tflite.BuiltinOperator.LESS_EQUAL,
                // tflite.BuiltinOperator.LOCAL_RESPONSE_NORMALIZATION,
                tflite.BuiltinOperator.LOG => VisitUnary(op, UnaryOp.Log),

                // tflite.BuiltinOperator.LOGICAL_AND,
                // tflite.BuiltinOperator.LOGICAL_NOT,
                // tflite.BuiltinOperator.LOGICAL_OR,
                tflite.BuiltinOperator.LOGISTIC => VisitLogistic(op),

                // tflite.BuiltinOperator.LOG_SOFTMAX,
                // tflite.BuiltinOperator.LSH_PROJECTION,
                // tflite.BuiltinOperator.LSTM,
                // tflite.BuiltinOperator.MATRIX_DIAG,
                // tflite.BuiltinOperator.MATRIX_SET_DIAG,
                tflite.BuiltinOperator.MAXIMUM => VisitBinary(op, BinaryOp.Max),

                // tflite.BuiltinOperator.MAX_POOL_2D,
                tflite.BuiltinOperator.MEAN => VisitReduce(op, ReduceOp.Mean, 0f),
                tflite.BuiltinOperator.MINIMUM => VisitBinary(op, BinaryOp.Min),

                tflite.BuiltinOperator.MIRROR_PAD => VisitMirrorPad(op),
                tflite.BuiltinOperator.MUL => VisitBinary(op, BinaryOp.Mul, op.BuiltinOptionsAsMulOptions().FusedActivationFunction),
                tflite.BuiltinOperator.NEG => VisitUnary(op, UnaryOp.Neg),

                // tflite.BuiltinOperator.NON_MAX_SUPPRESSION_V4,
                // tflite.BuiltinOperator.NON_MAX_SUPPRESSION_V5,
                // tflite.BuiltinOperator.NOT_EQUAL,
                // tflite.BuiltinOperator.ONE_HOT,
                tflite.BuiltinOperator.PACK => VisitPack(op),
                tflite.BuiltinOperator.PAD => VisitPad(op),
                tflite.BuiltinOperator.PADV2 => VisitPadV2(op),
                // tflite.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES,
                tflite.BuiltinOperator.POW => VisitBinary(op, BinaryOp.Pow),

                // tflite.BuiltinOperator.PRELU,
                // tflite.BuiltinOperator.QUANTIZE,
                // tflite.BuiltinOperator.RANGE,
                // tflite.BuiltinOperator.RANK,
                // tflite.BuiltinOperator.READ_VARIABLE,
                // tflite.BuiltinOperator.REAL,
                // tflite.BuiltinOperator.REDUCE_ALL,
                // tflite.BuiltinOperator.REDUCE_ANY,
                tflite.BuiltinOperator.REDUCE_MAX => VisitReduce(op, ReduceOp.Max, float.MinValue),
                tflite.BuiltinOperator.REDUCE_MIN => VisitReduce(op, ReduceOp.Min, float.MaxValue),
                // tflite.BuiltinOperator.REDUCE_PROD,
                // tflite.BuiltinOperator.RELU,
                // tflite.BuiltinOperator.RELU6,
                // tflite.BuiltinOperator.RELU_N1_TO_1,
                tflite.BuiltinOperator.RESHAPE => VisitReshape(op),
                // tflite.BuiltinOperator.RESIZE_BILINEAR,
                // tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR,
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
                // tflite.BuiltinOperator.SHAPE,
                tflite.BuiltinOperator.SIN => VisitUnary(op, UnaryOp.Sin),

                // tflite.BuiltinOperator.SKIP_GRAM,
                tflite.BuiltinOperator.SLICE => VisitSlice(op),
                // tflite.BuiltinOperator.SOFTMAX,
                // tflite.BuiltinOperator.SPACE_TO_BATCH_ND,
                // tflite.BuiltinOperator.SPACE_TO_DEPTH,
                // tflite.BuiltinOperator.SPARSE_TO_DENSE,
                // tflite.BuiltinOperator.SPLIT,
                // tflite.BuiltinOperator.SPLIT_V,
                tflite.BuiltinOperator.SQRT => VisitUnary(op, UnaryOp.Sqrt),
                tflite.BuiltinOperator.SQUARE => VisitUnary(op, UnaryOp.Square),

                // tflite.BuiltinOperator.SQUARED_DIFFERENCE,
                // tflite.BuiltinOperator.SQUEEZE,
                // tflite.BuiltinOperator.STRIDED_SLICE,
                tflite.BuiltinOperator.SUB => VisitBinary(op, BinaryOp.Sub, op.BuiltinOptionsAsSubOptions().FusedActivationFunction),

                tflite.BuiltinOperator.SUM => VisitReduce(op, ReduceOp.Sum, 0f),
                // tflite.BuiltinOperator.SVDF,
                tflite.BuiltinOperator.TANH => VisitUnary(op, UnaryOp.Tanh),

                // tflite.BuiltinOperator.TILE,
                // tflite.BuiltinOperator.TOPK_V2,
                tflite.BuiltinOperator.TRANSPOSE => VisitTranspose(op),
                // tflite.BuiltinOperator.TRANSPOSE_CONV,
                // tflite.BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM,
                // tflite.BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_RNN,
                // tflite.BuiltinOperator.UNIQUE,
                // tflite.BuiltinOperator.UNPACK,
                // tflite.BuiltinOperator.VAR_HANDLE,
                // tflite.BuiltinOperator.WHERE,
                // tflite.BuiltinOperator.WHILE,
                // tflite.BuiltinOperator.ZEROS_LIKE,
                _ => throw new NotSupportedException($"Unsupported tflite operator: {builtinCode}."),
            };

            if (output is Expr expr)
            {
                Debug.Assert(op.OutputsLength == 1, "Op outputs length should be 1.");
                _outputTensors.Add(op.Outputs(0), expr);
            }
            else if (output is IReadOnlyList<Expr> exprs)
            {
                Debug.Assert(op.OutputsLength == exprs.Count, $"Op outputs length should be {op.OutputsLength}.");
                for (int i = 0; i < op.OutputsLength; i++)
                {
                    _outputTensors.Add(op.Outputs(i), exprs[i]);
                }
            }
            else
            {
                throw new InvalidOperationException("Visit result is not expression(s).");
            }
        }

        private Expr GetInputExprs(in tflite.Operator op, int index)
        {
            var id = op.Inputs(index);

            if (_outputTensors.TryGetValue(id, out var expr))
            {
                return expr;
            }
            else
            {
                // Maybe constant
                var tensor = _subGraph.Tensors(id) ?? throw new InvalidDataException($"Cannot find tensor (id:{id}).");
                var buffer = _model.Buffers((int)tensor.Buffer) ?? throw new InvalidDataException($"Cannot find buffer (id:{tensor.Buffer}).");
                var data = buffer.GetDataBytes();
                if (!data.IsEmpty)
                {
                    var con = new Const(GetIRType(tensor.GetShapeBytes(), tensor.Type), data.ToArray());
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
            var tensorCopy = tensor;
            return Enumerable.Range(0, tensor.ShapeLength).Select(i => tensorCopy.Shape(i)).ToArray();
        }
        
        private static IEnumerable<int> GetShapeDataFromConst(Expr shape)
        {
            return ((Const)shape).ToTensor<int>().ToArray();
        }
    }
}
