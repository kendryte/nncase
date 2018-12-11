using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class FullyConnected : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public Tensor<float> Weights { get; }

        public Tensor<float> Bias { get; }

        public ActivationFunctionType FusedActivationFunction { get; }

        public FullyConnected(ReadOnlySpan<int> dimensions, Tensor<float> weights, Tensor<float> bias, ActivationFunctionType fusedActivationFunction)
        {
            FusedActivationFunction = fusedActivationFunction;
            Weights = weights;
            Bias = bias;

            Input = AddInput("input", dimensions);
            Output = AddOutput("output", new[] {
                dimensions[0],
                weights.Dimensions[0]
            });
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var input = context.TFOutputs[Input.Connection.From];
            var weights = Weights.ToHWIO();
            var bias = Bias.ToNHWC();

            var y = graph.Reshape(input, graph.Const(new[] { -1, Weights.Dimensions[1] }));
            y = graph.MatMul(y, graph.Const(weights));
            context.TFOutputs[Output] = graph.AddActivation(graph.BiasAdd(y, graph.Const(bias)), FusedActivationFunction);
        }
    }
}
