using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.Model;

namespace NnCase.Converter.K210.Model.Layers
{
    public class K210Upload : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public string Name { get; set; }

        public K210Upload(ReadOnlySpan<int> dimensions)
        {
            Input = AddInput("input", dimensions);
            Output = AddOutput("output", dimensions);
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            context.TFOutputs[Output] = context.TFGraph.Identity(context.TFOutputs[Input.Connection.From], Name);
        }
    }
}
