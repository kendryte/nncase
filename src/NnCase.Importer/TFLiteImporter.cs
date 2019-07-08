using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Text;
using MoreLinq;
using NnCase.IR;
using NnCase.IR.Operators;
using tflite;

namespace NnCase.Importer
{
    /// <summary>
    /// TFLite importer basic flow.
    /// </summary>
    public partial class TFLiteImporter
    {
        private readonly tflite.Model _model;
        private readonly tflite.SubGraph _subGraph;
        private readonly Graph _graph;
        private readonly Dictionary<BuiltinOperator, Action<Operator>> _lowerings;
        private readonly Dictionary<InputConnector, int> _inputTensors = new Dictionary<InputConnector, int>();
        private readonly Dictionary<int, OutputConnector> _outputTensors = new Dictionary<int, OutputConnector>();

        public TFLiteImporter(IServiceProvider serviceProvider, byte[] model, Graph graph)
        {
            _model = tflite.Model.GetRootAsModel(new FlatBuffers.ByteBuffer(model));
            _subGraph = _model.Subgraphs(0).Value;
            _graph = graph;

            _lowerings = new Dictionary<BuiltinOperator, Action<Operator>>
            {
                { BuiltinOperator.CONV_2D, ConvertConv2D },
                { BuiltinOperator.DEPTHWISE_CONV_2D, ConvertDepthwiseConv2D },
                { BuiltinOperator.MAX_POOL_2D, ConvertMaxPool2D },
                { BuiltinOperator.AVERAGE_POOL_2D, ConvertAveragePool2D },
                { BuiltinOperator.RESHAPE, ConvertReshape },
                { BuiltinOperator.ADD, ConvertAdd },
                { BuiltinOperator.SUB, ConvertSub },
                { BuiltinOperator.MUL, ConvertMul },
                { BuiltinOperator.DIV, ConvertDiv }
            };
        }

        public void Import()
        {
            Vector(_subGraph.Operators, _subGraph.OperatorsLength).ForEach(ConvertOperator);

            // connect tensors
            foreach (var input in _inputTensors)
            {
                if (_outputTensors.TryGetValue(input.Value, out var output))
                {
                    input.Key.Connect(output);
                }
                else
                {
                    var tensor = GetTensor(input.Value);
                    var buffer = _model.Buffers((int)tensor.Buffer).Value;
                    var data = buffer.GetDataSpan();

                    if (!data.IsEmpty)
                    {
                        var type = ToDataType(tensor.Type);
                        var shape = GetShape(tensor);
                        var constant = _graph.AddNode(new Constant(type, buffer.GetDataArray(), shape));
                        _outputTensors.Add(input.Value, constant.Output);
                        input.Key.Connect(constant.Output);
                    }
                }
            }

            // inputs
            foreach (var input in _inputTensors)
            {
                if (input.Key.Connection == null)
                {
                    // image
                    if (input.Key.Shape.Count == 4)
                    {
                        var inode = _graph.AddNode(new InputNode(input.Key.Type, ShapeUtility.NHWCToNCHW(input.Key.Shape), MemoryType.Main));
                        var surTrans = NCHWToNHWC(inode.Output.Type, inode.Output.Shape);
                        surTrans.Input.Connect(inode.Output);
                        input.Key.Connect(surTrans.Output);
                    }
                    else
                    {
                        var inode = _graph.AddNode(new InputNode(input.Key.Type, input.Key.Shape, MemoryType.Main));
                        input.Key.Connect(inode.Output);
                    }
                }
            }

            // outputs
            foreach (var output in _outputTensors)
            {
                if (output.Value.Connections.Count == 0)
                {
                    // image
                    if (output.Value.Shape.Count == 4)
                    {
                        var preTrans = NHWCToNCHW(output.Value.Type, output.Value.Shape);
                        var onode = _graph.AddNode(new OutputNode(preTrans.Output.Type, preTrans.Output.Shape));
                        preTrans.Input.Connect(output.Value);
                        onode.Input.Connect(preTrans.Output);
                    }
                    else
                    {
                        var onode = _graph.AddNode(new OutputNode(output.Value.Type, output.Value.Shape));
                        onode.Input.Connect(output.Value);
                    }
                }
            }
        }

        private void ConvertOperator(Operator op)
        {
            var opcode = _model.OperatorCodes((int)op.OpcodeIndex).Value;
            var builtinCode = opcode.BuiltinCode;

            if (_lowerings.TryGetValue(builtinCode, out var lower))
                lower(op);
            else
                throw new NotSupportedException($"TFLite operator is not supported: {builtinCode}");
        }

        private static IEnumerable<T> Vector<T>(Func<int, T?> getter, int count)
            where T : struct
        {
            for (int i = 0; i < count; i++)
            {
                var v = getter(i);
                if (v.HasValue)
                    yield return v.Value;
            }
        }

        private static IEnumerable<T> Vector<T>(Func<int, T> getter, int count)
            where T : struct
        {
            for (int i = 0; i < count; i++)
            {
                var v = getter(i);
                yield return v;
            }
        }

        private tflite.Tensor GetTensor(int index)
        {
            return _subGraph.Tensors(index).Value;
        }

        private DenseTensor<T> LoadTensor<T>(tflite.Tensor tensor)
            where T : unmanaged
        {
            var buffer = GetBuffer<T>(tensor);
            var shape = GetShape(tensor);
            var data = MemoryMarshal.Cast<byte, T>(buffer.GetDataSpan());
            return new DenseTensor<T>(data.ToArray(), shape);
        }

        private tflite.Buffer GetBuffer<T>(tflite.Tensor tensor)
            where T : unmanaged
        {
            var expectedType = ToTensorType<T>();
            var actualType = tensor.Type;
            if (expectedType != actualType)
                throw new InvalidOperationException($"Expect {expectedType} tensor but got {actualType}");

            var buffer = _model.Buffers((int)tensor.Buffer);
            if (!buffer.HasValue || buffer.Value.DataLength == 0)
                throw new InvalidOperationException("Cannot read tensor buffer");
            return buffer.Value;
        }

        private static tflite.TensorType ToTensorType<T>()
            where T : unmanaged
        {
            if (typeof(T) == typeof(float))
                return TensorType.FLOAT32;
            else if (typeof(T) == typeof(int))
                return TensorType.INT32;
            else
                throw new ArgumentException($"Invalid element type: {typeof(T).Name}");
        }

        private static DataType ToDataType(tflite.TensorType type)
        {
            switch (type)
            {
                case TensorType.FLOAT32:
                    return DataType.Float32;
                case TensorType.UINT8:
                    return DataType.UInt8;
                default:
                    throw new ArgumentException($"Unsupported tensor type: {type}");
            }
        }

        private static Shape GetShape(tflite.Tensor tensor)
        {
            return new Shape(tensor.GetShapeSpan());
        }

        private static ValueRange<float> ToFloatClampRange(tflite.ActivationFunctionType activationFunction)
        {
            switch (activationFunction)
            {
                case ActivationFunctionType.NONE:
                    return new ValueRange<float> { Min = float.MinValue, Max = float.MaxValue };
                case ActivationFunctionType.RELU:
                    return new ValueRange<float> { Min = 0, Max = float.MaxValue };
                case ActivationFunctionType.RELU_N1_TO_1:
                    return new ValueRange<float> { Min = -1, Max = 1 };
                case ActivationFunctionType.RELU6:
                    return new ValueRange<float> { Min = 0, Max = 6 };
                default:
                    throw new ArgumentOutOfRangeException($"Unsupported activation: {activationFunction}");
            }
        }
    }
}
