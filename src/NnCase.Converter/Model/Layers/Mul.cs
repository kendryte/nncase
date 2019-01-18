using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class Mul : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public Tensor<float> Scale { get; }

        public Mul(ReadOnlySpan<int> dimensions, Tensor<float> scale)
        {
            Input = AddInput("input", dimensions);
            Output = AddOutput("output", dimensions);
            Scale = scale;
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var input = context.TFOutputs[Input.Connection.From];

            context.TFOutputs[Output] = graph.Mul(input, graph.Const(Scale.ToNHWC()));
        }
    }
}
