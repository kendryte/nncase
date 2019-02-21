using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public enum Padding
    {
        Same,
        Valid
    }

    public enum ActivationFunctionType
    {
        Linear,
        Relu,
        Relu6
    }

    public class Conv2d : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public Tensor<float> Weights { get; }

        public Tensor<float> Bias { get; }

        public Padding Padding { get; }

        public int StrideWidth { get; }

        public int StrideHeight { get; }

        public ActivationFunctionType FusedActivationFunction { get; }

        public int KernelWidth => Weights.Dimensions[3];

        public int KernelHeight => Weights.Dimensions[2];

        public int InputChannels => Weights.Dimensions[1];

        public int OutputChannels => Weights.Dimensions[0];

        public Conv2d(ReadOnlySpan<int> dimensions, Tensor<float> weights, Tensor<float> bias, Padding padding, int strideWidth, int strideHeight, ActivationFunctionType fusedActivationFunction)
        {
            Padding = padding;
            StrideWidth = strideWidth;
            StrideHeight = strideHeight;
            FusedActivationFunction = fusedActivationFunction;
            Weights = weights;
            Bias = bias;

            Input = AddInput("input", dimensions);
            Output = AddOutput("output", new[] {
                dimensions[0],
                Weights.Dimensions[0],
                GetOutputSize(dimensions[2], KernelHeight, strideHeight, padding),
                GetOutputSize(dimensions[3], KernelWidth, strideWidth, padding)
            });
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var input = context.TFOutputs[Input.Connection.From];
            var weights = Weights.ToHWIO();
            var bias = Bias?.ToNHWC();

            var y = graph.Conv2D(input, graph.Const(weights),
                new long[] { 1, StrideHeight, StrideWidth, 1 }, Padding.ToString().ToUpperInvariant());
            if (bias != null)
                y = graph.BiasAdd(y, graph.Const(bias));
            context.TFOutputs[Output] = graph.AddActivation(y, FusedActivationFunction);
        }
    }
}
