using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class LeakyRelu : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public float Slope { get; }

        public LeakyRelu(ReadOnlySpan<int> dimensions, float slope)
        {
            Input = AddInput("input", dimensions);
            Output = AddOutput("output", dimensions);
            Slope = slope;
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var x = context.TFOutputs[Input.Connection.From];
            context.TFOutputs[Output] = graph.Maximum(x, graph.Mul(x, graph.Const(Slope)));
        }
    }
}
