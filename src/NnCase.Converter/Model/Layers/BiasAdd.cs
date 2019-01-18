using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class BiasAdd : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public Tensor<float> Bias { get; }

        public BiasAdd(ReadOnlySpan<int> dimensions, Tensor<float> bias)
        {
            Input = AddInput("input", dimensions);
            Output = AddOutput("output", dimensions);
            Bias = bias;
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var input = context.TFOutputs[Input.Connection.From];

            context.TFOutputs[Output] = graph.BiasAdd(input, graph.Const(Bias.ToNHWC()));
        }
    }
}
