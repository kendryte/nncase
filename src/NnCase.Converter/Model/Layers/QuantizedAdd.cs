using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class QuantizedAdd : Layer
    {
        public InputConnector InputA { get; }

        public InputConnector InputB { get; }

        public OutputConnector Output { get; }

        public QuantizedAdd(ReadOnlySpan<int> aDimensions, ReadOnlySpan<int> bDimensions)
        {
            InputA = AddInput("inputA", aDimensions);
            InputB = AddInput("inputB", bDimensions);
            Output = AddOutput("output", aDimensions);
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var a = context.TFOutputs[InputA.Connection.From];
            var b = context.TFOutputs[InputB.Connection.From];

            context.TFOutputs[Output] = graph.Add(a, b);
        }
    }
}
