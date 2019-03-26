using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class Mean : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public Mean(ReadOnlySpan<int> dimensions)
        {
            Input = AddInput("input", dimensions);
            Output = AddOutput("output", new[] {
                dimensions[0],
                dimensions[1]
            });
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var input = context.TFOutputs[Input.Connection.From];

            context.TFOutputs[Output] = graph.Mean(input, graph.Const(new[] { 1, 2 }));
        }
    }
}
