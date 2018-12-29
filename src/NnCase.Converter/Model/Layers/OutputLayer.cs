using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class OutputLayer : Layer
    {
        public InputConnector Input { get; }

        public string Name { get; set; }

        public OutputLayer(ReadOnlySpan<int> dimensions)
        {
            Input = AddInput("input", dimensions);
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            context.Outputs[this] = context.TFGraph.Identity(context.TFOutputs[Input.Connection.From], Name);
        }
    }
}
