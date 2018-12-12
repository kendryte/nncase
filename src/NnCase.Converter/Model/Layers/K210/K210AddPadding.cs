using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.Model.Layers.K210
{
    public class K210AddPadding : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public K210AddPadding(ReadOnlySpan<int> dimensions)
        {
            Input = AddInput("input", dimensions);
            Output = AddOutput("output", new[] {
                dimensions[0],
                dimensions[1],
                4,
                4
            });
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var input = context.TFOutputs[Input.Connection.From];

            context.TFOutputs[Output] = graph.Pad(input, graph.Const(new[,] { { 0, 0 }, { 0, 3 }, { 0, 3 }, { 0, 0 } }));
        }
    }
}
