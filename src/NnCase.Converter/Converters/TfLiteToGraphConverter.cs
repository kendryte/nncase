using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Text;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.Converters
{
    public class TfLiteToGraphConverter
    {
        private readonly tflite.Model _model;
        private readonly tflite.SubGraph _graph;

        private readonly Dictionary<InputConnector, int> _inputs;
        private readonly Dictionary<int, OutputConnector> _outputs;

        public Graph Graph { get; private set; }

        public TfLiteToGraphConverter(tflite.Model model, tflite.SubGraph graph)
        {
            _model = model;
            _graph = graph;

            _inputs = new Dictionary<InputConnector, int>();
            _outputs = new Dictionary<int, OutputConnector>();
        }

        public void Convert()
        {
            var layers = _graph.GetOperators().Select(ConvertOperator).ToList();
            foreach (var inputPair in _inputs)
            {
                if (_outputs.TryGetValue(inputPair.Value, out var output))
                {
                    inputPair.Key.SetConnection(output);
                }
            }

            var inputs = new List<InputLayer>();
            foreach (var conn in _inputs.Keys.Where(o => o.Connection == null))
            {
                var input = new InputLayer(conn.Dimensions);
                conn.SetConnection(input.Output);
                inputs.Add(input);
            }

            var outputs = new List<OutputLayer>();
            foreach (var conn in _outputs.Values.Where(o => !o.Connections.Any()))
            {
                var output = new OutputLayer(conn.Dimensions);
                conn.AddConnection(output.Input);
                outputs.Add(output);
            }

            Graph = new Graph(inputs, outputs);
        }

        private Layer ConvertOperator(tflite.Operator op)
        {
            var opCode = _model.OperatorCodes((int)op.OpcodeIndex).Value;
            switch (opCode.BuiltinCode)
            {
                case tflite.BuiltinOperator.SPACE_TO_BATCH_ND:
                    return ConvertSpaceToBatchNd(op);
                case tflite.BuiltinOperator.CONV_2D:
                    return ConvertConv2d(op);
                case tflite.BuiltinOperator.DEPTHWISE_CONV_2D:
                    return ConvertDepthwiseConv2d(op);
                case tflite.BuiltinOperator.AVERAGE_POOL_2D:
                    return ConvertAveragePool2d(op);
                case tflite.BuiltinOperator.L2_NORMALIZATION:
                    return ConvertL2Normalization(op);
                case tflite.BuiltinOperator.ADD:
                    return ConvertAdd(op);
                case tflite.BuiltinOperator.MUL:
                    return ConvertMul(op);
                case tflite.BuiltinOperator.FULLY_CONNECTED:
                    return ConvertFullyConnected(op);
                case tflite.BuiltinOperator.MAX_POOL_2D:
                    return ConvertMaxPool2d(op);
                case tflite.BuiltinOperator.SOFTMAX:
                    return ConvertSoftmax(op);
                case tflite.BuiltinOperator.CONCATENATION:
                    return ConvertConcatenation(op);
                case tflite.BuiltinOperator.MAXIMUM:
                    return ConvertMaximum(op);
                case tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
                    return ConvertResizeNearestNeighbor(op);
                case tflite.BuiltinOperator.LEAKY_RELU:
                    return ConvertLeakyRelu(op);
                case tflite.BuiltinOperator.MEAN:
                    return ConvertMean(op);
                case tflite.BuiltinOperator.RESHAPE:
                    return ConvertReshape(op);
                case tflite.BuiltinOperator.PAD:
                    return ConvertPad(op);
                case tflite.BuiltinOperator.LOGISTIC:
                    return ConvertLogistic(op);
                default:
                    throw new LayerNotSupportedException(opCode.BuiltinCode.ToString());
            }
        }

        private Layer ConvertSpaceToBatchNd(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var input = _graph.Tensors(inputs[0]).Value;
            var blockShape = _graph.Tensors(inputs[1]).Value;
            var paddings = _graph.Tensors(inputs[2]).Value;

            var layer = new SpaceToBatchNd(input.GetShapeArray().ToNCHW(), _model.GetTensor<int>(blockShape), _model.GetTensor<int>(paddings));
            _inputs.Add(layer.Input, inputs[0]);
            _outputs.Add(op.Outputs(0), layer.Output);
            return layer;
        }

        private Layer ConvertPad(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var input = _graph.Tensors(inputs[0]).Value;
            var paddings = _graph.Tensors(inputs[1]).Value;

            var layer = new Pad(input.GetShapeArray().ToNCHW(), _model.GetTensor<int>(paddings));
            if (!layer.Paddings.ToArray().SequenceEqual(new[] { 0, 0, 1, 1, 1, 1, 0, 0 }))
                throw new LayerNotSupportedException("Pad", "Only paddings of [[0,0],[1,1],[1,1],[0,0]] is supported");
            _inputs.Add(layer.Input, inputs[0]);
            _outputs.Add(op.Outputs(0), layer.Output);
            return layer;
        }

        private Layer ConvertConv2d(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var input = _graph.Tensors(inputs[0]).Value;
            var options = op.BuiltinOptions<tflite.Conv2DOptions>().Value;
            var weights = _graph.Tensors(inputs[1]).Value;
            var bias = _graph.Tensors(inputs[2]).Value;

            var layer = new Conv2d(input.GetShapeArray().ToNCHW(), _model.GetTensor<float>(weights).ToOIHW(), _model.GetTensor<float>(bias),
                options.Padding.ToPadding(), options.StrideW, options.StrideH, options.FusedActivationFunction.ToActivationFunction());
            _inputs.Add(layer.Input, inputs[0]);
            _outputs.Add(op.Outputs(0), layer.Output);
            return layer;
        }

        private Layer ConvertDepthwiseConv2d(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var input = _graph.Tensors(inputs[0]).Value;
            var options = op.BuiltinOptions<tflite.DepthwiseConv2DOptions>().Value;
            var weights = _graph.Tensors(inputs[1]).Value;
            var bias = _graph.Tensors(inputs[2]).Value;
            var depthMul = options.DepthMultiplier;

            if (input.GetShapeArray().ToNCHW()[1] == 1 && depthMul != 1)
            {
                var layer = new Conv2d(input.GetShapeArray().ToNCHW(), _model.GetTensor<float>(weights).ToOIHW().Transpose(new[] { 1, 0, 2, 3 }), _model.GetTensor<float>(bias),
                    options.Padding.ToPadding(), options.StrideW, options.StrideH, options.FusedActivationFunction.ToActivationFunction());
                _inputs.Add(layer.Input, inputs[0]);
                _outputs.Add(op.Outputs(0), layer.Output);
                return layer;
            }
            else if (depthMul != 1)
            {
                throw new LayerNotSupportedException("DEPTHWISE_CONV_2D", "depth_multiplier must be 1");
            }

            {
                var layer = new DepthwiseConv2d(input.GetShapeArray().ToNCHW(), _model.GetTensor<float>(weights).ToOIHW(), _model.GetTensor<float>(bias),
                    options.Padding.ToPadding(), options.StrideW, options.StrideH, options.FusedActivationFunction.ToActivationFunction());
                _inputs.Add(layer.Input, inputs[0]);
                _outputs.Add(op.Outputs(0), layer.Output);
                return layer;
            }
        }

        private Layer ConvertAveragePool2d(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var input = _graph.Tensors(inputs[0]).Value;
            var options = op.BuiltinOptions<tflite.Pool2DOptions>().Value;

            var layer = new AveragePool2d(input.GetShapeArray().ToNCHW(), options.Padding.ToPadding(), options.FilterWidth, options.FilterHeight, options.StrideW,
                options.StrideH, options.FusedActivationFunction.ToActivationFunction());
            _inputs.Add(layer.Input, inputs[0]);
            _outputs.Add(op.Outputs(0), layer.Output);
            return layer;
        }

        private Layer ConvertReshape(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var input = _graph.Tensors(inputs[0]).Value;
            var options = op.BuiltinOptions<tflite.ReshapeOptions>().Value;

            var layer = new TensorflowReshape(input.GetShapeArray().ToNCHW(), options.GetNewShapeArray().ToNCHW());
            _inputs.Add(layer.Input, inputs[0]);
            _outputs.Add(op.Outputs(0), layer.Output);
            return layer;
        }

        private Layer ConvertL2Normalization(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var input = _graph.Tensors(inputs[0]).Value;

            if (input.ShapeLength == 4 && (input.Shape(1) != 1 || input.Shape(2) != 1))
            {
                var flatten = new TensorflowFlatten(input.GetShapeArray().ToNCHW());
                var layer = new L2Normalization(flatten.Output.Dimensions);
                layer.Input.SetConnection(flatten.Output);
                _inputs.Add(flatten.Input, inputs[0]);
                _outputs.Add(op.Outputs(0), layer.Output);
                return layer;
            }
            else
            {
                var layer = new L2Normalization(input.GetShapeArray().ToNCHW());
                _inputs.Add(layer.Input, inputs[0]);
                _outputs.Add(op.Outputs(0), layer.Output);
                return layer;
            }
        }

        private Layer ConvertAdd(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var inputA = _graph.Tensors(inputs[0]).Value;
            var inputB = _graph.Tensors(inputs[1]).Value;

            var layer = new Add(inputA.GetShapeArray().ToNCHW(), inputB.GetShapeArray().ToNCHW());
            _inputs.Add(layer.InputA, inputs[0]);
            _inputs.Add(layer.InputB, inputs[1]);
            _outputs.Add(op.Outputs(0), layer.Output);
            return layer;
        }

        private Layer ConvertMul(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var inputA = _graph.Tensors(inputs[0]).Value;
            var inputB = _graph.Tensors(inputs[1]).Value;

            if (inputA.ShapeLength == 0)
            {
                var layer = new Mul(inputB.GetShapeArray().ToNCHW(), _model.GetScalar<float>(inputA));
                _inputs.Add(layer.Input, inputs[1]);
                _outputs.Add(op.Outputs(0), layer.Output);
                return layer;
            }
            else if (inputB.ShapeLength == 0)
            {
                var layer = new Mul(inputA.GetShapeArray().ToNCHW(), _model.GetScalar<float>(inputB));
                _inputs.Add(layer.Input, inputs[0]);
                _outputs.Add(op.Outputs(0), layer.Output);
                return layer;
            }
            else
            {
                throw new LayerNotSupportedException(op.ToString(), "Only scalar multiply is supported");
            }
        }

        private Layer ConvertFullyConnected(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var input = _graph.Tensors(inputs[0]).Value;
            var options = op.BuiltinOptions<tflite.FullyConnectedOptions>().Value;
            var weights = _graph.Tensors(inputs[1]).Value;
            var bias = _graph.Tensors(inputs[2]).Value;

            if (input.ShapeLength == 4 && (input.Shape(1) != 1 || input.Shape(2) != 1))
            {
                var flatten = new TensorflowFlatten(input.GetShapeArray().ToNCHW());
                var layer = new FullyConnected(flatten.Output.Dimensions, _model.GetTensor<float>(weights), _model.GetTensor<float>(bias),
                    options.FusedActivationFunction.ToActivationFunction());
                layer.Input.SetConnection(flatten.Output);
                _inputs.Add(flatten.Input, inputs[0]);
                _outputs.Add(op.Outputs(0), layer.Output);
                return layer;
            }
            else
            {
                var layer = new FullyConnected(input.GetShapeArray().ToNCHW(), _model.GetTensor<float>(weights), _model.GetTensor<float>(bias),
                    options.FusedActivationFunction.ToActivationFunction());
                _inputs.Add(layer.Input, inputs[0]);
                _outputs.Add(op.Outputs(0), layer.Output);
                return layer;
            }
        }

        private Layer ConvertMaxPool2d(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var input = _graph.Tensors(inputs[0]).Value;
            var options = op.BuiltinOptions<tflite.Pool2DOptions>().Value;

            var layer = new MaxPool2d(input.GetShapeArray().ToNCHW(), options.Padding.ToPadding(), options.FilterWidth, options.FilterHeight, options.StrideW,
                options.StrideH, options.FusedActivationFunction.ToActivationFunction());
            _inputs.Add(layer.Input, inputs[0]);
            _outputs.Add(op.Outputs(0), layer.Output);
            return layer;
        }

        private Layer ConvertSoftmax(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var input = _graph.Tensors(inputs[0]).Value;
            var options = op.BuiltinOptions<tflite.SoftmaxOptions>().Value;

            var layer = new Softmax(input.GetShapeArray().ToNCHW());
            _inputs.Add(layer.Input, inputs[0]);
            _outputs.Add(op.Outputs(0), layer.Output);
            return layer;
        }

        private Layer ConvertLogistic(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var input = _graph.Tensors(inputs[0]).Value;

            var layer = new Logistic(input.GetShapeArray().ToNCHW());
            _inputs.Add(layer.Input, inputs[0]);
            _outputs.Add(op.Outputs(0), layer.Output);
            return layer;
        }

        private Layer ConvertConcatenation(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var options = op.BuiltinOptions<tflite.ConcatenationOptions>().Value;
            if (options.Axis != 3)
                throw new NotSupportedException("Axis of concatenation must be 3.");
            var layer = new Concatenation(inputs.Select(x => new ReadOnlyMemory<int>(_graph.Tensors(x).Value.GetShapeArray().ToNCHW())));
            for (int i = 0; i < inputs.Length; i++)
                _inputs.Add(layer.Inputs[i], inputs[i]);
            _outputs.Add(op.Outputs(0), layer.Output);
            return layer;
        }

        private Layer ConvertMaximum(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var inputA = _graph.Tensors(inputs[0]).Value;
            var inputB = _graph.Tensors(inputs[1]).Value;

            var layer = new Maximum(inputA.GetShapeArray().ToNCHW(), inputB.GetShapeArray().ToNCHW());
            _inputs.Add(layer.InputA, inputs[0]);
            _inputs.Add(layer.InputB, inputs[1]);
            _outputs.Add(op.Outputs(0), layer.Output);
            return layer;
        }

        private Layer ConvertResizeNearestNeighbor(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var input = _graph.Tensors(inputs[0]).Value;
            var newSize = _model.GetTensor<int>(_graph.Tensors(inputs[1]).Value);
            var options = op.BuiltinOptions<tflite.ResizeNearestNeighborOptions>().Value;

            var blockShape = _graph.Tensors(inputs[1]).Value;
            var layer = new ResizeNearestNeighbor(input.GetShapeArray().ToNCHW(), newSize[1], newSize[0], options.AlignCorners);
            _inputs.Add(layer.Input, inputs[0]);
            _outputs.Add(op.Outputs(0), layer.Output);
            return layer;
        }

        private Layer ConvertLeakyRelu(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var input = _graph.Tensors(inputs[0]).Value;
            var options = op.BuiltinOptions<tflite.LeakyReluOptions>().Value;

            var layer = new LeakyRelu(input.GetShapeArray().ToNCHW(), options.Alpha);
            _inputs.Add(layer.Input, inputs[0]);
            _outputs.Add(op.Outputs(0), layer.Output);
            return layer;
        }

        private Layer ConvertMean(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var input = _graph.Tensors(inputs[0]).Value;
            var axes = _model.GetTensor<int>(_graph.Tensors(inputs[1]).Value);
            var output = _graph.Tensors(op.GetOutputsArray()[0]).Value;

            if (axes.ToArray().SequenceEqual(new[] { 1, 2 }))
            {
                var layer = new GlobalAveragePool(input.GetShapeArray().ToNCHW());
                _inputs.Add(layer.Input, inputs[0]);
                var reshape = new Reshape(layer.Output.Dimensions, output.GetShapeArray().ToNCHW());
                reshape.Input.SetConnection(layer.Output);
                _outputs.Add(op.Outputs(0), reshape.Output);
                return reshape;
            }
            else
            {
                throw new LayerNotSupportedException(op.ToString(), "Only [1,2] axis mean is supported");
            }
        }
    }

    static class TfLiteExtensions
    {
        public static IEnumerable<tflite.Operator> GetOperators(this tflite.SubGraph subGraph)
        {
            for (int i = 0; i < subGraph.OperatorsLength; i++)
                yield return subGraph.Operators(i).Value;
        }

        public static Tensor<T> GetTensor<T>(this tflite.Model model, tflite.Tensor tensor)
            where T : unmanaged
        {
            if (typeof(T) == typeof(float) && tensor.Type != tflite.TensorType.FLOAT32)
                throw new InvalidOperationException($"expect FLOAT32 tensor but got {tensor.Type}, use '--inference_type=FLOAT' when converting via toco.");

            var buffer = model.Buffers((int)tensor.Buffer).Value;
            return new DenseTensor<T>(MemoryMarshal.Cast<byte, T>(buffer.GetDataBytes()).ToArray(), tensor.GetShapeArray());
        }

        public static T GetScalar<T>(this tflite.Model model, tflite.Tensor tensor)
            where T : unmanaged
        {
            if (tensor.ShapeLength != 0)
                throw new InvalidOperationException("Tensor is not a scalar");
            var buffer = model.Buffers((int)tensor.Buffer).Value;
            return MemoryMarshal.Cast<byte, T>(buffer.GetDataBytes())[0];
        }

        public static int[] ToNCHW(this int[] shape)
        {
            if (shape.Length == 2)
                return shape;
            return new[] { shape[0], shape[3], shape[1], shape[2] };
        }

        public static int[] ToNHWC(this int[] shape)
        {
            if (shape.Length == 2)
                return shape;
            return new[] { shape[0], shape[2], shape[3], shape[1] };
        }

        public static int[] ToNC(this int[] shape)
        {
            if (shape.Length == 2)
                return shape;
            else
                return new[] { shape[0], shape[3] * shape[1] * shape[2] };
        }

        public static Tensor<T> ToOIHW<T>(this Tensor<T> weights)
        {
            return weights.Transpose<T>(new[] { 0, 3, 1, 2 });
        }

        public static Padding ToPadding(this tflite.Padding padding)
        {
            switch (padding)
            {
                case tflite.Padding.SAME:
                    return Padding.Same;
                case tflite.Padding.VALID:
                    return Padding.Valid;
                default:
                    throw new ArgumentOutOfRangeException(nameof(padding));
            }
        }

        public static ActivationFunctionType ToActivationFunction(this tflite.ActivationFunctionType activation)
        {
            switch (activation)
            {
                case tflite.ActivationFunctionType.NONE:
                    return ActivationFunctionType.Linear;
                case tflite.ActivationFunctionType.RELU:
                    return ActivationFunctionType.Relu;
                case tflite.ActivationFunctionType.RELU6:
                    return ActivationFunctionType.Relu6;
                default:
                    throw new NotSupportedException(nameof(activation));
            }
        }
    }
}
