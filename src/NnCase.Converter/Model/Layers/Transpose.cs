using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class Transpose : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        private readonly int[] _axes;
        public ReadOnlySpan<int> Axes => _axes;

        public Transpose(ReadOnlySpan<int> dimensions, ReadOnlySpan<int> axes)
        {
            var newDims = new int[dimensions.Length];
            for (int i = 0; i < axes.Length; i++)
                newDims[i] = dimensions[axes[i]];

            Input = AddInput("input", dimensions);
            Output = AddOutput("output", newDims);
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var input = context.TFOutputs[Input.Connection.From];
            var graph = context.TFGraph;

            context.TFOutputs[Output] = graph.Transpose(input, graph.Const(Axes.ToTFAxes()));
        }
    }
}
