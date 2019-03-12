using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class PriorBox : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public float[] MinSizes { get; }

        public float[] MaxSizes { get; }

        public PriorBox(ReadOnlySpan<int> dimensions, int imageWidth, int imageHeight, float[] minSizes, float[] maxSizes)
        {
            Input = AddInput("input", dimensions);
            //Output = AddOutput("output", new[] { dimensions[0], dimensions[1], outputHeight, outputWidth });
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var input = context.TFOutputs[Input.Connection.From];

        }
    }
}
