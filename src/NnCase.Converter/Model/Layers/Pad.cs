using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class Pad : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public Tensor<int> Paddings { get; }

        public Pad(ReadOnlySpan<int> dimensions, Tensor<int> paddings)
        {
            Paddings = paddings;

            Input = AddInput("input", dimensions);
            Output = AddOutput("output", new[] { dimensions[0], dimensions[1], dimensions[2] + 2, dimensions[3] + 2 });
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var input = context.TFOutputs[Input.Connection.From];

            context.TFOutputs[Output] = graph.Pad(input, graph.Const(Paddings.ToTFTensor()));
        }
    }
}
