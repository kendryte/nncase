using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.Model;

namespace NnCase.Converter.K210.Model.Layers
{
    public class K210RemovePadding : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public K210RemovePadding(ReadOnlySpan<int> dimensions)
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

            var y = graph.MaxPool(input, new long[] { 1, 1, 1, 1 }, new long[] { 1, 4, 4, 1 }, "VALID");
            context.TFOutputs[Output] = graph.Reshape(y, graph.Const(new[] { -1, Input.Dimensions[1] }));
        }
    }
}
