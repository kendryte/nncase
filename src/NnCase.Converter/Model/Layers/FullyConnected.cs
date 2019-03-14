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
            var y = context.TFOutputs[Input.Connection.From];
            var weights = Weights.ToHWIO();

            if (Input.Dimensions.Length == 4 && Input.Dimensions[2] == 1 && Input.Dimensions[3] == 1)
                y = graph.Reshape(y, graph.Const(new[] { Input.Dimensions[0], Input.Dimensions[1] }));

            y = graph.MatMul(y, graph.Const(weights));
            if (Bias != null)
                y = graph.BiasAdd(y, graph.Const(Bias.ToNHWC()));

            context.TFOutputs[Output] = graph.AddActivation(y, FusedActivationFunction);
        }
    }
}
