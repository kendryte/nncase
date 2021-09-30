// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
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
            var created_inputs = new List<Var>();
            foreach (var inputId in _subGraph.GetInputsBytes())
            {
                var tensor = _subGraph.Tensors(inputId)!.Value;
                var input = new Var(tensor.Name, GetIRType(tensor.GetShapeBytes(), tensor.Type));
                created_inputs.Add(input);
                _outputTensors.Add(inputId, input);
            }

            // 2. Convert ops
            for (int i = 0; i < _subGraph.OperatorsLength; i++)
            {
                var op = _subGraph.Operators(i)!.Value;
                Visit(op);
            }

            var module = new Module();
            return module;
        }

        /// <summary>
        /// Create IR type from tflite shape and tensor type.
        /// </summary>
        /// <param name="shape">Shape.</param>
        /// <param name="type">Tensor type.</param>
        /// <returns>Created IR type.</returns>
        private static IRType GetIRType(Span<int> shape, tflite.TensorType type)
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
                tflite.BuiltinOperator.ABS => VisitBinary(op, BinaryOp.Add, tflite.ActivationFunctionType.NONE),
                _ => new NotSupportedException($"Unsupported tflite operator: {opcode}."),
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

        private Expr GetInputExprs(in tflite.Operator op, int index) =>
            _outputTensors[op.Inputs(index)];

        private (Expr Expr0, Expr Expr1) GetInputExprs(in tflite.Operator op, int index0, int index1) =>
            (_outputTensors[op.Inputs(index0)], _outputTensors[op.Inputs(index1)]);
    }
}
