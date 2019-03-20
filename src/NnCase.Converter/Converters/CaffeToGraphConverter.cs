using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using System.Threading.Tasks;
using Caffe;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.Converters
{
    public class CaffeToGraphConverter
    {
        private readonly Caffe.NetParameter _model;
        private readonly Dictionary<string, OutputConnector> _outputs;

        public Graph Graph { get; private set; }

        public CaffeToGraphConverter(Caffe.NetParameter model)
        {
            _model = model;

            _outputs = new Dictionary<string, OutputConnector>();
        }

        public void Convert()
        {
            var layers = _model.Layer.Select(ConvertLayer).Where(x => x != null).ToList();

            var inputs = new List<InputLayer>(layers.OfType<InputLayer>());

            var outputs = new List<OutputLayer>();
            int i = 0;
            foreach (var conn in _outputs.Values.Where(o => !o.Connections.Any()))
            {
                var output = new OutputLayer(conn.Dimensions) { Name = $"output_{i++}" };
                conn.AddConnection(output.Input);
                outputs.Add(output);
            }

            Graph = new Graph(inputs, outputs);
        }

        private Layer ConvertLayer(LayerParameter layerParam)
        {
            if (layerParam.Bottom.Any(x => x == "label"))
                return null;

            switch (layerParam.Type)
            {
                case "Input":
                    return ConvertInput(layerParam);
                case "Split":
                    return ConvertSplit(layerParam);
                case "Convolution":
                case "DepthwiseConvolution":
                    return ConvertConvolution(layerParam);
                case "BatchNorm":
                    return ConvertBatchNorm(layerParam);
                case "Scale":
                    return ConvertScale(layerParam);
                case "ReLU":
                    return ConvertReLU(layerParam);
                case "PReLU":
                    return ConvertPReLU(layerParam);
                case "Pooling":
                    return ConvertPooling(layerParam);
                case "Softmax":
                    return ConvertSoftmax(layerParam);
                case "Eltwise":
                    return ConvertEltwise(layerParam);
                case "InnerProduct":
                    return ConvertInnerProduct(layerParam);
                case "L2Normalization":
                    return ConvertL2Normalization(layerParam);
                case "Normalize":
                    return ConvertNormalize(layerParam);
                default:
                    throw new LayerNotSupportedException(layerParam.Type);
            }
        }

        private Layer ConvertInput(LayerParameter layerParam)
        {
            var dim = layerParam.InputParam.Shape[0].Dim.Select(x => (int)x).ToArray();
            dim[0] = 1;
            var layer = new InputLayer(dim) { Name = layerParam.Name };

            var weights = new float[,] { { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 } }.ToTensor().Reshape(new[] { 3, 3, 1, 1 });
            var conv2d = new Conv2d(layer.Output.Dimensions, weights, null, Padding.Valid, 1, 1, ActivationFunctionType.Linear);
            conv2d.Input.SetConnection(layer.Output);

            _outputs.Add(layerParam.Top[0], conv2d.Output);
            return layer;
        }

        private Layer ConvertSplit(LayerParameter layerParam)
        {
            var input = _outputs[layerParam.Bottom[0]];
            var layer = new Identity(input.Dimensions) { Name = layerParam.Name };
            layer.Input.SetConnection(input);
            foreach (var output in layerParam.Top)
                _outputs.Add(output, layer.Output);
            return layer;
        }

        private Layer ConvertConvolution(LayerParameter layerParam)
        {
            var input = _outputs[layerParam.Bottom[0]];
            var param = layerParam.ConvolutionParam;

            uint[] GetDefault(Google.Protobuf.Collections.RepeatedField<uint> field)
            {
                if (field.Count == 0)
                    return new uint[] { 1, 1 };
                else if (field.Count == 1)
                    return new[] { field[0], field[0] };
                else
                    return field.ToArray();
            }

            var padding = GetDefault(param.Pad);
            var strides = GetDefault(param.Stride);
            var kernelSize = param.KernelSize.Count == 1 ? new[] { param.KernelSize[0], param.KernelSize[0] }
                : (param.KernelSize.Count == 0 ? new[] { param.KernelH, param.KernelW } : param.KernelSize.ToArray());
            var group = param.Group == 0 ? 1 : param.Group;

            if (group == 1)
            {
                var weights = LoadBlob(layerParam.Blobs[0]);
                Conv2d conv2d;
                if (padding[0] != 0 || padding[1] != 0)
                {
                    if (padding[0] == 1 && padding[1] == 1 && strides[0] == 1 && strides[1] == 1)
                    {
                        conv2d = new Conv2d(input.Dimensions, weights, null, Padding.Same, 1, 1, ActivationFunctionType.Linear);
                        conv2d.Input.SetConnection(input);
                    }
                    else
                    {
                        var space = new SpaceToBatchNd(input.Dimensions, new[] { 1, 1 }.ToTensor(), new[,] { { (int)padding[0], (int)padding[0] }, { (int)padding[1], (int)padding[1], } }.ToTensor());
                        conv2d = new Conv2d(space.Output.Dimensions, weights, null, Padding.Valid, (int)strides[1], (int)strides[0], ActivationFunctionType.Linear);
                        space.Input.SetConnection(input);
                        conv2d.Input.SetConnection(space.Output);
                    }
                }
                else
                {
                    conv2d = new Conv2d(input.Dimensions, weights, null, Padding.Valid, (int)strides[1], (int)strides[0], ActivationFunctionType.Linear);
                    conv2d.Input.SetConnection(input);
                }

                _outputs.Add(layerParam.Top[0], conv2d.Output);
                return conv2d;
            }
            else if (group == param.NumOutput)
            {
                var weights = LoadBlob(layerParam.Blobs[0]).Transpose(new[] { 1, 0, 2, 3 });
                DepthwiseConv2d dwConv2d;

                if (padding[0] != 0 || padding[1] != 0)
                {
                    if (padding[0] == 1 && padding[1] == 1 && strides[0] == 1 && strides[1] == 1)
                    {
                        dwConv2d = new DepthwiseConv2d(input.Dimensions, weights, null, Padding.Same, 1, 1, ActivationFunctionType.Linear);
                        dwConv2d.Input.SetConnection(input);
                    }
                    else
                    {
                        var space = new SpaceToBatchNd(input.Dimensions, new[] { 1, 1 }.ToTensor(), new[,] { { (int)padding[0], (int)padding[0] }, { (int)padding[1], (int)padding[1], } }.ToTensor());
                        dwConv2d = new DepthwiseConv2d(space.Output.Dimensions, weights, null, Padding.Valid, (int)strides[1], (int)strides[0], ActivationFunctionType.Linear);
                        space.Input.SetConnection(input);
                        dwConv2d.Input.SetConnection(space.Output);
                    }
                }
                else
                {
                    dwConv2d = new DepthwiseConv2d(input.Dimensions, weights, null, Padding.Valid, (int)strides[1], (int)strides[0], ActivationFunctionType.Linear);
                    dwConv2d.Input.SetConnection(input);
                }

                _outputs.Add(layerParam.Top[0], dwConv2d.Output);
                return dwConv2d;
            }
            else
            {
                throw new NotSupportedException("Grouped conv2d is not supported.");
            }
        }

        private Layer ConvertBatchNorm(LayerParameter layerParam)
        {
            var input = _outputs[layerParam.Bottom[0]];
            var param = layerParam.BatchNormParam;
            var eps = param == null || param.Eps == 0 ? 1e-5f : param.Eps;

            var scaleFactor = LoadBlob(layerParam.Blobs[2])[0];
            var mean = LoadBlob(layerParam.Blobs[0], scaleFactor);
            var variance = LoadBlob(layerParam.Blobs[1], scaleFactor);
            var scale = Enumerable.Repeat(1.0f, mean.Dimensions[0]).ToArray().ToTensor();
            var offset = Enumerable.Repeat(0.0f, mean.Dimensions[0]).ToArray().ToTensor();
            var layer = new BatchNormalization(input.Dimensions, scale, offset, mean, variance, eps);
            layer.Input.SetConnection(input);
            _outputs[layerParam.Top[0]] = layer.Output;
            return layer;
        }

        private Layer ConvertScale(LayerParameter layerParam)
        {
            var input = _outputs[layerParam.Bottom[0]];
            var param = layerParam.ScaleParam;

            var scale = LoadBlob(layerParam.Blobs[0]);
            var bias = LoadBlob(layerParam.Blobs[1]);
            var mul = new Mul(input.Dimensions, scale);
            mul.Input.SetConnection(input);
            var biasAdd = new BiasAdd(mul.Output.Dimensions, bias);
            biasAdd.Input.SetConnection(mul.Output);
            _outputs[layerParam.Top[0]] = biasAdd.Output;
            return biasAdd;
        }

        private Layer ConvertReLU(LayerParameter layerParam)
        {
            var input = _outputs[layerParam.Bottom[0]];
            var param = layerParam.ReluParam;

            if (param != null && param.NegativeSlope != 0)
                throw new NotSupportedException("Non zero negative slope of relu is not supported.");
            var layer = new Relu(input.Dimensions);
            layer.Input.SetConnection(input);
            _outputs[layerParam.Top[0]] = layer.Output;
            return layer;
        }

        private Layer ConvertPReLU(LayerParameter layerParam)
        {
            var input = _outputs[layerParam.Bottom[0]];
            var param = layerParam.ReluParam;

            var slope = LoadBlob(layerParam.Blobs[0]);
            if (param != null && param.NegativeSlope != 0)
                throw new NotSupportedException();
            var layer = new PRelu(input.Dimensions, slope);
            layer.Input.SetConnection(input);
            _outputs[layerParam.Top[0]] = layer.Output;
            return layer;
        }

        private Layer ConvertPooling(LayerParameter layerParam)
        {
            var input = _outputs[layerParam.Bottom[0]];
            var param = layerParam.PoolingParam;

            var ksizes = GetCaffeSize(param.KernelH, param.KernelW, param.KernelSize);
            var strides = GetCaffeSize(param.StrideH, param.StrideW, param.Stride);
            var paddings = GetCaffeSize(param.PadH, param.PadW, param.Pad);

            if (param.GlobalPooling)
            {
                ksizes[0] = (uint)input.Dimensions[2];
                ksizes[1] = (uint)input.Dimensions[3];
                strides[0] = strides[1] = 1;
                paddings[0] = paddings[1] = 0;
            }

            if (paddings[0] != paddings[1] || (paddings[0] != 0 && paddings[0] != 1))
                throw new NotSupportedException("Custom paddings are not supprted.");

            Padding padding = paddings[0] == 0 ? Padding.Valid : Padding.Same;

            if (param.Pool == PoolingParameter.Types.PoolMethod.Ave)
            {
                var layer = new AveragePool2d(input.Dimensions, padding, (int)ksizes[1], (int)ksizes[0], (int)strides[1], (int)strides[0], ActivationFunctionType.Linear);
                layer.Input.SetConnection(input);
                _outputs[layerParam.Top[0]] = layer.Output;
                return layer;
            }
            else if (param.Pool == PoolingParameter.Types.PoolMethod.Max)
            {
                var layer = new MaxPool2d(input.Dimensions, padding, (int)ksizes[1], (int)ksizes[0], (int)strides[1], (int)strides[0], ActivationFunctionType.Linear);
                layer.Input.SetConnection(input);
                _outputs[layerParam.Top[0]] = layer.Output;
                return layer;
            }
            else
                throw new LayerNotSupportedException(param.Pool.ToString());
        }

        private uint[] GetCaffeSize(uint h, uint w, uint size)
        {
            var sizes = new uint[2];
            if(size == 0)
            {
                sizes[0] = h;
                sizes[1] = w;
            }
            else
            {
                sizes[0] = sizes[1] = size;
            }

            return sizes;
        }

        private Layer ConvertSoftmax(LayerParameter layerParam)
        {
            var input = _outputs[layerParam.Bottom[0]];
            var param = layerParam.SoftmaxParam;

            var layer = new Softmax(input.Dimensions);
            layer.Input.SetConnection(input);
            _outputs.Add(layerParam.Top[0], layer.Output);
            return layer;
        }

        private Layer ConvertEltwise(LayerParameter layerParam)
        {
            var a = _outputs[layerParam.Bottom[0]];
            var b = _outputs[layerParam.Bottom[1]];
            var param = layerParam.EltwiseParam;

            var layer = new Add(a.Dimensions, b.Dimensions);
            layer.InputA.SetConnection(a);
            layer.InputB.SetConnection(b);
            _outputs[layerParam.Top[0]] = layer.Output;
            return layer;
        }

        private Layer ConvertInnerProduct(LayerParameter layerParam)
        {
            var input = _outputs[layerParam.Bottom[0]];
            var param = layerParam.InnerProductParam;

            var weights = LoadBlob(layerParam.Blobs[0]);

            if (input.Dimensions.Length == 4 && (input.Dimensions[2] != 1 || input.Dimensions[3] != 1))
            {
                var flatten = new Reshape(input.Dimensions, new[] { -1, input.Dimensions.GetSize() });
                var layer = new FullyConnected(flatten.Output.Dimensions, weights, null, ActivationFunctionType.Linear);
                flatten.Input.SetConnection(input);
                layer.Input.SetConnection(flatten.Output);
                _outputs[layerParam.Top[0]] = layer.Output;
                return layer;
            }
            else
            {
                var layer = new FullyConnected(input.Dimensions, weights, null, ActivationFunctionType.Linear);
                layer.Input.SetConnection(input);
                _outputs[layerParam.Top[0]] = layer.Output;
                return layer;
            }
        }

        private Layer ConvertL2Normalization(LayerParameter layerParam)
        {
            var input = _outputs[layerParam.Bottom[0]];
            var param = layerParam.L2NormalizationParam;

            var layer = new L2Normalization(input.Dimensions);
            layer.Input.SetConnection(input);
            _outputs[layerParam.Top[0]] = layer.Output;
            return layer;
        }

        private Layer ConvertNormalize(LayerParameter layerParam)
        {
            var input = _outputs[layerParam.Bottom[0]];
            var param = layerParam.NormalizeParam;

            var layer = new L2Normalization(input.Dimensions);
            layer.Input.SetConnection(input);
            _outputs[layerParam.Top[0]] = layer.Output;
            return layer;
        }

        private Tensor<float> LoadBlob(BlobProto blob, float scale = 1.0f)
        {
            var tensor = new DenseTensor<float>(blob.Shape.Dim.Select(x => (int)x).ToArray());
            var span = tensor.Buffer.Span;
            for (int i = 0; i < span.Length; i++)
                span[i] = blob.Data[i] / scale;
            return tensor;
        }
    }
}
