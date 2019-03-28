using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using TensorFlow;

namespace NnCase.Converter.K210.Model.Layers
{
    public enum K210PoolType
    {
        None = 0,
        MaxPool2x2 = 1,
        AveragePool2x2 = 2,
        MaxPool4x4 = 3,
        AveragePool4x4 = 4,
        LeftTop = 5,
        AveragePool2x2Stride1 = 8,
        MaxPool2x2Stride1 = 9
    }

    public enum K210Conv2dType
    {
        Conv2d,
        DepthwiseConv2d = 1
    }

    public class K210Conv2d : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public Guid OutputBeforeActivation { get; } = Guid.NewGuid();

        public K210Conv2dType Conv2dType { get; }

        public Tensor<float> Weights { get; }

        public Tensor<float> Bias { get; }

        public K210PoolType PoolType { get; }

        public ActivationFunctionType FusedActivationFunction { get; }

        public Layer NonTrivialActivation { get; }

        public int KernelWidth => Weights.Dimensions[3];

        public int KernelHeight => Weights.Dimensions[2];

        public int InputChannels => Weights.Dimensions[1];

        public int OutputChannels => Conv2dType == K210Conv2dType.Conv2d ? Weights.Dimensions[0] : InputChannels;

        public bool IsChannelwiseOutput { get; set; }

        public K210Conv2d(ReadOnlySpan<int> dimensions, K210Conv2dType conv2dType, Tensor<float> weights, Tensor<float> bias, K210PoolType poolType, ActivationFunctionType fusedActivationFunction, Layer nonTrivalActivation)
        {
            if (conv2dType == K210Conv2dType.DepthwiseConv2d && poolType != K210PoolType.None)
                throw new ArgumentOutOfRangeException("Downsampling is not supported in dwConv2d.");
            if (dimensions[2] < 4 || dimensions[3] < 4)
                throw new ArgumentOutOfRangeException("Lower than 4x4 input is not supported in dwConv2d.");
            if (nonTrivalActivation != null && fusedActivationFunction != ActivationFunctionType.Linear)
                throw new ArgumentException("There should be only one activation in conv2d.");

            Conv2dType = conv2dType;
            PoolType = poolType;
            FusedActivationFunction = fusedActivationFunction;
            NonTrivialActivation = nonTrivalActivation;
            Weights = weights;
            Bias = bias;

            var stride = GetStride();

            if (dimensions[2] / stride < 4 || dimensions[3] / stride < 4)
                throw new ArgumentOutOfRangeException("Lower than 4x4 output is not supported in dwConv2d.");

            Input = AddInput("input", dimensions);
            Output = AddOutput("output", new[] {
                dimensions[0],
                OutputChannels,
                dimensions[2] / stride,
                dimensions[3] / stride
            });
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var input = context.TFOutputs[Input.Connection.From];
            var weights = Weights.ToHWIO();
            var bias = Bias.ToNHWC();

            TFOutput y;
            if (PoolType == K210PoolType.LeftTop)
            {
                y = KernelWidth == 1 ? input : graph.SpaceToBatchND(input, graph.Const(new[] { 1, 1 }), graph.Const(new[,] { { 1, 1 }, { 1, 1 } }));
                y = Conv2dType == K210Conv2dType.Conv2d
                    ? graph.Conv2D(y, graph.Const(weights), new long[] { 1, 2, 2, 1 }, "VALID")
                    : graph.DepthwiseConv2dNative(y, graph.Const(weights), new long[] { 1, 2, 2, 1 }, "VALID");
            }
            else
            {
                y = Conv2dType == K210Conv2dType.Conv2d
                    ? graph.Conv2D(input, graph.Const(weights), new long[] { 1, 1, 1, 1 }, "SAME")
                    : graph.DepthwiseConv2dNative(input, graph.Const(weights), new long[] { 1, 1, 1, 1 }, "SAME");
            }

            y = graph.BiasAdd(y, graph.Const(bias));
            context.AdditionalTFOutputs[OutputBeforeActivation] = y;
            y = AddActivation(graph, y, FusedActivationFunction, NonTrivialActivation);

            switch (PoolType)
            {
                case K210PoolType.MaxPool2x2:
                    y = graph.MaxPool(y, new long[] { 1, 2, 2, 1 }, new long[] { 1, 2, 2, 1 }, "VALID");
                    break;
                case K210PoolType.AveragePool2x2:
                    y = graph.AvgPool(y, new long[] { 1, 2, 2, 1 }, new long[] { 1, 2, 2, 1 }, "VALID");
                    break;
                case K210PoolType.MaxPool4x4:
                    y = graph.MaxPool(y, new long[] { 1, 4, 4, 1 }, new long[] { 1, 4, 4, 1 }, "VALID");
                    break;
                case K210PoolType.AveragePool4x4:
                    y = graph.AvgPool(y, new long[] { 1, 4, 4, 1 }, new long[] { 1, 4, 4, 1 }, "VALID");
                    break;
                case K210PoolType.MaxPool2x2Stride1:
                    y = graph.MaxPool(y, new long[] { 1, 2, 2, 1 }, new long[] { 1, 1, 1, 1 }, "SAME");
                    break;
                case K210PoolType.AveragePool2x2Stride1:
                    y = graph.AvgPool(y, new long[] { 1, 2, 2, 1 }, new long[] { 1, 1, 1, 1 }, "SAME");
                    break;
                default:
                    break;
            }

            context.TFOutputs[Output] = y;
        }

        private TFOutput AddActivation(TFGraph graph, TFOutput y, ActivationFunctionType fusedActivationFunction, Layer nonTrivialActivation)
        {
            if (fusedActivationFunction != ActivationFunctionType.Linear)
            {
                y = graph.AddActivation(y, fusedActivationFunction);
            }
            else if (nonTrivialActivation != null)
            {
                if (nonTrivialActivation is LeakyRelu leakyRelu)
                {
                    y = graph.Maximum(y, graph.Mul(y, graph.Const(leakyRelu.Slope)));
                }
            }

            return y;
        }

        private int GetStride()
        {
            int stride;
            switch (PoolType)
            {
                case K210PoolType.None:
                case K210PoolType.MaxPool2x2Stride1:
                case K210PoolType.AveragePool2x2Stride1:
                    stride = 1;
                    break;
                case K210PoolType.LeftTop:
                case K210PoolType.MaxPool2x2:
                case K210PoolType.AveragePool2x2:
                    stride = 2;
                    break;
                case K210PoolType.MaxPool4x4:
                case K210PoolType.AveragePool4x4:
                    stride = 4;
                    break;
                default:
                    throw new NotSupportedException();
            }

            return stride;
        }
    }
}
