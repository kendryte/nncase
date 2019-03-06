using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class ResizeBilinear : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public bool AlignCorners { get; }

        public ResizeBilinear(ReadOnlySpan<int> dimensions, int outputWidth, int outputHeight, bool alignCorners = false)
        {
            AlignCorners = alignCorners;
            Input = AddInput("input", dimensions);
            Output = AddOutput("output", new[] { dimensions[0], dimensions[1], outputHeight, outputWidth });
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var input = context.TFOutputs[Input.Connection.From];

            context.TFOutputs[Output] = graph.ResizeBilinear(input, graph.Const(new long[] { Output.Dimensions[2], Output.Dimensions[3] }), AlignCorners);
        }
    }
}
