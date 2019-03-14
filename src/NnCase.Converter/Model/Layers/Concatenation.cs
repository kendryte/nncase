using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class Concatenation : Layer
    {
        public IReadOnlyList<InputConnector> Inputs { get; }

        public OutputConnector Output { get; }

        public Concatenation(IEnumerable<ReadOnlyMemory<int>> dimensions)
        {
            int i = 0;
            var inputs = new List<InputConnector>();
            var outputDims = dimensions.First().ToArray();
            outputDims[1] = 0;
            foreach (var dimension in dimensions)
            {
                inputs.Add(AddInput("input" + i++, dimension.Span));
                outputDims[1] += dimension.Span[1];
            }

            Inputs = inputs;
            Output = AddOutput("output", outputDims);
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var inputs = Inputs.Select(x=> context.TFOutputs[x.Connection.From]).ToArray();

            if (inputs.Length == 1)
                context.TFOutputs[Output] = inputs[0];
            else
                context.TFOutputs[Output] = graph.Concat(graph.Const(3), inputs);
        }
    }
}
