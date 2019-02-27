using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class GlobalAveragePool : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public GlobalAveragePool(ReadOnlySpan<int> dimensions)
        {
            Input = AddInput("input", dimensions);
            Output = AddOutput("output", new[] {
                dimensions[0],
                dimensions[1],
                1,
                1
            });
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var input = context.TFOutputs[Input.Connection.From];

            context.TFOutputs[Output] = graph.AvgPool(input, new long[] { 1, Input.Dimensions[2], Input.Dimensions[3], 1 },
                new long[] { 1, 1, 1, 1 }, "VALID");
        }
    }
}
