using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using paddle = Paddle.Framework.Proto;

namespace NnCase.Converter.Converters
{
    public class PaddleToGraphConverter
    {
        private readonly string _modelPath;
        private readonly paddle.ProgramDesc _programDesc;
        private readonly Dictionary<InputConnector, string> _inputs;
        private readonly Dictionary<string, OutputConnector> _outputs;

        private paddle.BlockDesc _subgraph;
        public Graph Graph { get; private set; }

        public PaddleToGraphConverter(string modelPath)
        {
            _modelPath = modelPath;
            _programDesc = paddle.ProgramDesc.Parser.ParseFrom(File.ReadAllBytes(Path.Combine(modelPath, "__model__")));

            _inputs = new Dictionary<InputConnector, string>();
            _outputs = new Dictionary<string, OutputConnector>();
        }

        public void Convert(int subgraphIndex)
        {
            _subgraph = _programDesc.Blocks[subgraphIndex];
            var layers = _subgraph.Ops.Select(ConvertOperator).Where(x => x != null).ToList();
            foreach (var inputPair in _inputs)
            {
                if (_outputs.TryGetValue(inputPair.Value, out var output))
                {
                    inputPair.Key.SetConnection(output);
                }
            }

            var inputs = new List<InputLayer>(layers.OfType<InputLayer>());
            foreach (var conn in _inputs.Keys.Where(o => o.Connection == null))
            {
                if (_inputs.TryGetValue(conn, out var varName))
                {
                    var vv = GetVar(varName);
                    var v = LoadVarDataOrDefault<float>(varName);
                    var c = new Constant(GetVarShape(varName), v);
                    conn.SetConnection(c.Output);
                }
                else
                {
                    var input = new InputLayer(conn.Dimensions);
                    conn.SetConnection(input.Output);
                    inputs.Add(input);
                }
            }

            var outputs = new List<OutputLayer>(layers.OfType<OutputLayer>());
            foreach (var conn in _outputs.Values.Where(o => !o.Connections.Any()))
            {
                var output = new OutputLayer(conn.Dimensions);
                conn.AddConnection(output.Input);
                outputs.Add(output);
            }

            Graph = new Graph(inputs, outputs);
        }

        private Layer ConvertOperator(paddle.OpDesc op)
        {
            switch (op.Type)
            {
                case "feed":
                    return ConvertFeed(op);
                case "fetch":
                    return ConvertFetch(op);
                case "conv2d":
                case "depthwise_conv2d":
                    return ConvertConv2d(op);
                case "elementwise_add":
                    return ConvertElementwiseAdd(op);
                case "batch_norm":
                    return ConvertBatchNorm(op);
                case "relu":
                    return ConvertRelu(op);
                case "relu6":
                    return ConvertRelu6(op);
                case "pool2d":
                    return ConvertPool2d(op);
                case "reshape":
                    return ConvertReshape(op);
                case "softmax":
                    return ConvertSoftmax(op);
                case "bilinear_interp":
                    return ConvertBilinearInterp(op);
                case "nearest_interp":
                    return ConvertNearestInterp(op);
                case "prior_box":
                    return ConvertPriorBox(op);
                case "transpose2":
                    return ConvertTranspose2(op);
                case "reshape2":
                    return ConvertShape2(op);
                case "concat":
                    return ConvertConcat(op);
                case "scale":
                    return ConvertScale(op);
                case "mul":
                    return ConvertMul(op);
                case "assign_value":
                case "shape":
                case "slice":
                case "cast":
                case "fill_constant":
                case "elementwise_mul":
                    return null;
                default:
                    throw new LayerNotSupportedException(op.Type);
            }
        }

        private Layer ConvertFeed(paddle.OpDesc op)
        {
            var output = GetParameter(op.Outputs, "Out").Arguments[0];
            var layer = new InputLayer(GetVarShape(output)) { Name = output };
            _outputs.Add(output, layer.Output);
            return layer;
        }

        private Layer ConvertConv2d(paddle.OpDesc op)
        {
            var padding = GetAttr(op, "paddings").Ints;
            var strides = GetAttr(op, "strides").Ints.ToArray();
            var groups = GetAttr(op, "groups").I;
            if (strides[0] == 0) strides[0] = 1;
            if (strides[1] == 0) strides[1] = 1;

            var input = GetParameter(op.Inputs, "Input").Arguments[0];
            var weights = GetParameter(op.Inputs, "Filter").Arguments[0];
            var weightsShape = GetVarShape(weights);
            var kernelWidth = weightsShape[3];
            var kernelHeight = weightsShape[2];
            var output = GetParameter(op.Outputs, "Output").Arguments[0];

            if (groups == 1)
            {
                Conv2d conv2d;
                if (padding[0] == 1 && padding[1] == 1 && strides[0] == 2 && strides[1] == 2 &&
                    kernelWidth == 3 && kernelHeight == 3)
                {
                    var space = new SpaceToBatchNd(GetVarShape(input), new[] { 1, 1 }.ToTensor(), new[,] { { 1, 1 }, { 1, 1, } }.ToTensor());
                    conv2d = new Conv2d(space.Output.Dimensions, LoadVarData<float>(weights), null, Padding.Valid, strides[1], strides[0], ActivationFunctionType.Linear);
                    conv2d.Input.SetConnection(space.Output);
                    _inputs.Add(space.Input, input);
                }
                else
                {
                    if (padding[0] != padding[1] || (padding[0] != 0 && padding[0] != 1))
                        throw new NotSupportedException();

                    conv2d = new Conv2d(GetVarShape(input), LoadVarData<float>(weights), null, Padding.Same, strides[1], strides[0], ActivationFunctionType.Linear);
                    _inputs.Add(conv2d.Input, input);
                }

                _outputs.Add(output, conv2d.Output);
                return conv2d;
            }
            else if (groups == weightsShape[0])
            {
                var w = LoadVarData<float>(weights).Transpose(new[] { 1, 0, 2, 3 });
                DepthwiseConv2d dwConv2d;
                if (padding[0] == 1 && padding[1] == 1 && strides[0] == 2 && strides[1] == 2 &&
                    kernelWidth == 3 && kernelHeight == 3)
                {
                    var space = new SpaceToBatchNd(GetVarShape(input), new[] { 1, 1 }.ToTensor(), new[,] { { 1, 1 }, { 1, 1, } }.ToTensor());
                    dwConv2d = new DepthwiseConv2d(space.Output.Dimensions, w, null, Padding.Valid, strides[1], strides[0], ActivationFunctionType.Linear);
                    dwConv2d.Input.SetConnection(space.Output);
                    _inputs.Add(space.Input, input);
                }
                else
                {
                    if (padding[0] != padding[1] || (padding[0] != 0 && padding[0] != 1))
                        throw new NotSupportedException();

                    dwConv2d = new DepthwiseConv2d(GetVarShape(input), w, null, Padding.Same, strides[1], strides[0], ActivationFunctionType.Linear);
                    _inputs.Add(dwConv2d.Input, input);
                }

                _outputs.Add(output, dwConv2d.Output);
                return dwConv2d;
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        private Layer ConvertElementwiseAdd(paddle.OpDesc op)
        {
            var x = GetParameter(op.Inputs, "X").Arguments[0];
            var y = GetParameter(op.Inputs, "Y").Arguments[0];
            var output = GetParameter(op.Outputs, "Out").Arguments[0];

            var layer = new Add(GetVarShape(x), GetVarShape(y));
            _inputs.Add(layer.InputA, x);
            _inputs.Add(layer.InputB, y);
            _outputs.Add(output, layer.Output);
            return layer;
        }

        private Layer ConvertScale(paddle.OpDesc op)
        {
            var x = GetParameter(op.Inputs, "X").Arguments[0];
            var scale = GetAttr(op, "scale").F;
            var output = GetParameter(op.Outputs, "Out").Arguments[0];

            var layer = new Mul(GetVarShape(x), scale);
            _inputs.Add(layer.Input, x);
            _outputs.Add(output, layer.Output);
            return layer;
        }

        private Layer ConvertMul(paddle.OpDesc op)
        {
            var x = GetParameter(op.Inputs, "X").Arguments[0];
            var y = GetParameter(op.Inputs, "Y").Arguments[0];
            var output = GetParameter(op.Outputs, "Out").Arguments[0];

            var layer = new FullyConnected(GetVarShape(x), LoadVarData<float>(y).Transpose(new[] { 1, 0 }), null, ActivationFunctionType.Linear);
            _inputs.Add(layer.Input, x);
            _outputs.Add(output, layer.Output);
            return layer;
        }

        private Layer ConvertBatchNorm(paddle.OpDesc op)
        {
            var epsilon = GetAttr(op, "epsilon").F;
            var offset = GetParameter(op.Inputs, "Bias").Arguments[0];
            var mean = GetParameter(op.Inputs, "Mean").Arguments[0];
            var scale = GetParameter(op.Inputs, "Scale").Arguments[0];
            var variance = GetParameter(op.Inputs, "Variance").Arguments[0];
            var x = GetParameter(op.Inputs, "X").Arguments[0];
            var output = GetParameter(op.Outputs, "Y").Arguments[0];

            var layer = new BatchNormalization(GetVarShape(x), LoadVarData<float>(scale), LoadVarData<float>(offset),
                LoadVarData<float>(mean), LoadVarData<float>(variance), epsilon);
            _inputs.Add(layer.Input, x);
            _outputs.Add(output, layer.Output);
            return layer;
        }

        private Layer ConvertRelu(paddle.OpDesc op)
        {
            var x = GetParameter(op.Inputs, "X").Arguments[0];
            var output = GetParameter(op.Outputs, "Out").Arguments[0];

            var layer = new Relu(GetVarShape(x));
            layer.Input.SetConnection(_outputs[x]);
            _outputs[output] = layer.Output;
            return layer;
        }

        private Layer ConvertRelu6(paddle.OpDesc op)
        {
            var x = GetParameter(op.Inputs, "X").Arguments[0];
            var output = GetParameter(op.Outputs, "Out").Arguments[0];

            var layer = new Relu6(GetVarShape(x));
            layer.Input.SetConnection(_outputs[x]);
            _outputs[output] = layer.Output;
            return layer;
        }

        private Layer ConvertPool2d(paddle.OpDesc op)
        {
            var type = GetAttr(op, "pooling_type").S;
            var ksize = GetAttr(op, "ksize").Ints;
            var strides = GetAttr(op, "strides").Ints;
            var x = op.Inputs[0].Arguments[0];
            var output = op.Outputs[0].Arguments[0];

            if (GetAttr(op, "global_pooling").B)
            {
                var shape = GetVarShape(x);
                ksize[0] = shape[2];
                ksize[1] = shape[3];
            }

            if (type == "avg")
            {
                var layer = new AveragePool2d(GetVarShape(x), Padding.Valid, ksize[1], ksize[0], strides[1], strides[0], ActivationFunctionType.Linear);
                _inputs.Add(layer.Input, x);
                _outputs.Add(output, layer.Output);
                return layer;
            }
            else if (type == "max")
            {
                var layer = new MaxPool2d(GetVarShape(x), Padding.Valid, ksize[1], ksize[0], strides[1], strides[0], ActivationFunctionType.Linear);
                _inputs.Add(layer.Input, x);
                _outputs.Add(output, layer.Output);
                return layer;
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        private Layer ConvertReshape(paddle.OpDesc op)
        {
            var shape = GetAttr(op, "shape").Ints;
            var x = GetParameter(op.Inputs, "X").Arguments[0];
            var output = GetParameter(op.Outputs, "Out").Arguments[0];

            var layer = new Reshape(GetVarShape(x), shape.ToArray());
            _inputs.Add(layer.Input, x);
            _outputs.Add(output, layer.Output);
            return layer;
        }

        private Layer ConvertSoftmax(paddle.OpDesc op)
        {
            var x = GetParameter(op.Inputs, "X").Arguments[0];
            var output = GetParameter(op.Outputs, "Out").Arguments[0];

            var layer = new Softmax(GetVarShape(x));
            _inputs.Add(layer.Input, x);
            _outputs.Add(output, layer.Output);
            return layer;
        }

        private Layer ConvertBilinearInterp(paddle.OpDesc op)
        {
            var w = GetAttr(op, "out_w").I;
            var h = GetAttr(op, "out_h").I;
            var alignCorners = GetAttr(op, "align_corners").B;
            var x = GetParameter(op.Inputs, "X").Arguments[0];
            var output = GetParameter(op.Outputs, "Out").Arguments[0];

            var layer = new ResizeBilinear(GetVarShape(x), w, h, alignCorners);
            _inputs.Add(layer.Input, x);
            _outputs.Add(output, layer.Output);
            return layer;
        }

        private Layer ConvertNearestInterp(paddle.OpDesc op)
        {
            var w = GetAttr(op, "out_w").I;
            var h = GetAttr(op, "out_h").I;
            var alignCorners = GetAttr(op, "align_corners").B;
            var x = GetParameter(op.Inputs, "X").Arguments[0];
            var output = GetParameter(op.Outputs, "Out").Arguments[0];

            var layer = new ResizeNearestNeighbor(GetVarShape(x), w, h, alignCorners);
            _inputs.Add(layer.Input, x);
            _outputs.Add(output, layer.Output);
            return layer;
        }

        private Layer ConvertPriorBox(paddle.OpDesc op)
        {
            var image = GetParameter(op.Inputs, "Image").Arguments[0];
            var imageShape = GetVarShape(image);

            var minSizes = GetAttr(op, "min_sizes").Floats.ToArray();
            var maxSizes = GetAttr(op, "max_sizes").Floats.ToArray();
            var aspectRatios = GetAttr(op, "aspect_ratios").Floats.ToArray();
            var variances = GetAttr(op, "variances").Floats.ToArray();
            var flip = GetAttr(op, "flip").B;
            var clip = GetAttr(op, "clip").B;
            var stepWidth = GetAttr(op, "step_w").I;
            var stepHeight = GetAttr(op, "step_h").I;
            var offset = GetAttr(op, "offset").F;

            var input = GetParameter(op.Inputs, "Input").Arguments[0];
            var boxes = GetParameter(op.Outputs, "Boxes").Arguments[0];
            var v = GetParameter(op.Outputs, "Variances").Arguments[0];

            var layer = new PriorBox(GetVarShape(input), imageShape[3], imageShape[2], minSizes, maxSizes, aspectRatios, variances, flip, clip, stepWidth, stepHeight, offset);
            _inputs.Add(layer.Input, input);
            _outputs.Add(boxes, layer.Boxes);
            _outputs.Add(v, layer.VariancesOutput);
            return layer;
        }

        private Layer ConvertTranspose2(paddle.OpDesc op)
        {
            var x = GetParameter(op.Inputs, "X").Arguments[0];
            var output = GetParameter(op.Outputs, "Out").Arguments[0];
            var axes = GetAttr(op, "axis").Ints.ToArray();

            var layer = new Transpose(GetVarShape(x), axes);
            _inputs.Add(layer.Input, x);
            _outputs.Add(output, layer.Output);
            return layer;
        }

        private Layer ConvertShape2(paddle.OpDesc op)
        {
            var x = GetParameter(op.Inputs, "X").Arguments[0];
            var output = GetParameter(op.Outputs, "Out").Arguments[0];
            var shape = GetAttr(op, "shape").Ints.ToArray();

            var layer = new Reshape(GetVarShape(x), shape);
            _inputs.Add(layer.Input, x);
            _outputs.Add(output, layer.Output);
            return layer;
        }

        private Layer ConvertConcat(paddle.OpDesc op)
        {
            var x = GetParameter(op.Inputs, "X").Arguments;
            var output = GetParameter(op.Outputs, "Out").Arguments[0];

            var layer = new Concatenation(x.Select(x => new ReadOnlyMemory<int>(GetVarShape(x))));
            for (int i = 0; i < x.Count; i++)
                _inputs.Add(layer.Inputs[i], x[i]);
            _outputs.Add(output, layer.Output);
            return layer;
        }

        private Layer ConvertFetch(paddle.OpDesc op)
        {
            var input = op.Inputs[0].Arguments[0];
            var layer = new OutputLayer(GetVarShape(input)) { Name = input };
            _inputs.Add(layer.Input, input);
            return layer;
        }

        private paddle.VarDesc GetVar(string name)
        {
            return _subgraph.Vars.First(o => o.Name == name);
        }

        private paddle.OpDesc.Types.Var GetParameter(IEnumerable<paddle.OpDesc.Types.Var> vars, string name)
        {
            return vars.First(o => o.Parameter == name);
        }

        private int[] GetVarShape(string name)
        {
            var v = GetVar(name);
            return v.Type.LodTensor.Tensor.Dims.Select(x => (int)x).ToArray();
        }

        private paddle.OpDesc.Types.Attr GetAttr(paddle.OpDesc op, string name)
        {
            return op.Attrs.First(x => x.Name == name);
        }

        private Tensor<T> LoadVarData<T>(string name)
            where T : unmanaged
        {
            var v = GetVar(name);
            var reader = new SpanReader(File.ReadAllBytes(Path.Combine(_modelPath, name)));

            var version = reader.Read<uint>();
            var lodLevel = reader.Read<ulong>();
            for (uint i = 0; i < lodLevel; i++)
            {
                var len = reader.Read<ulong>();
                reader.Skip((int)len);
            }

            version = reader.Read<uint>();
            if (version != 0)
                throw new NotSupportedException();
            {
                var descSize = reader.Read<int>();
                reader.Skip(descSize);
            }

            var rest = reader.ReadAsSpan();
            var data = MemoryMarshal.Cast<byte, T>(rest);

            return new DenseTensor<T>(data.ToArray(), GetVarShape(name));
        }

        private Tensor<T> LoadVarDataOrDefault<T>(string name)
            where T : unmanaged
        {
            if (File.Exists(Path.Combine(_modelPath, name)))
            {
                return LoadVarData<T>(name);
            }
            else
            {
                var v = GetVar(name);
                return new DenseTensor<T>(v.Type.LodTensor.Tensor.Dims.Select(x => (int)x).ToArray());
            }
        }
    }
}
