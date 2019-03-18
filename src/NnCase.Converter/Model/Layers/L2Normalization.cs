using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace NnCase.Converter.Model.Layers
{
    public class L2Normalization : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public L2Normalization(ReadOnlySpan<int> dimensions)
        {
            Input = AddInput("input", dimensions);
            Output = AddOutput("output", dimensions);
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var x = context.TFOutputs[Input.Connection.From];
            var y = graph.Sum(graph.Square(x), graph.Const(Input.Dimensions.Length - 1), keep_dims: true);
            y = graph.Rsqrt(graph.Maximum(y, graph.Const(1e-10f)));

            context.TFOutputs[Output] = graph.Mul(x, y);
        }
    }
}
