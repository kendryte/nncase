using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class Softmax : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public Softmax(ReadOnlySpan<int> dimensions)
        {
            Input = AddInput("input", dimensions);
            Output = AddOutput("output", dimensions);
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var x = context.TFOutputs[Input.Connection.From];
            context.TFOutputs[Output] = graph.Softmax(x);
        }
    }
}
