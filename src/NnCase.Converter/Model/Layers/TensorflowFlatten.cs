using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class TensorflowFlatten : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public TensorflowFlatten(ReadOnlySpan<int> dimensions)
        {
            Input = AddInput("input", dimensions);
            Output = AddOutput("output", new[] { dimensions[0], dimensions.GetSize() / dimensions[0] });
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var input = context.TFOutputs[Input.Connection.From];
            var graph = context.TFGraph;

            context.TFOutputs[Output] = graph.Reshape(input, graph.Const(Output.Dimensions.ToArray()));
        }
    }
}
