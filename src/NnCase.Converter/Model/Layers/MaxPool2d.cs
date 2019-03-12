using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class MaxPool2d : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public Padding Padding { get; }

        public int StrideWidth { get; }

        public int StrideHeight { get; }

        public int FilterWidth { get; }

        public int FilterHeight { get; }

        public ActivationFunctionType FusedActivationFunction { get; }

        public MaxPool2d(ReadOnlySpan<int> dimensions, Padding padding, int filterWidth, int filterHeight, int strideWidth, int strideHeight, ActivationFunctionType fusedActivationFunction)
        {
            Padding = padding;
            StrideWidth = strideWidth;
            StrideHeight = strideHeight;
            FilterWidth = filterWidth;
            FilterHeight = filterHeight;
            FusedActivationFunction = fusedActivationFunction;

            Input = AddInput("input", dimensions);
            Output = AddOutput("output", new[] {
                dimensions[0],
                dimensions[1],
                GetOutputSize(dimensions[2], filterHeight, strideHeight, padding),
                GetOutputSize(dimensions[3], filterWidth, strideWidth, padding)
            });
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var input = context.TFOutputs[Input.Connection.From];

            context.TFOutputs[Output] = graph.AddActivation(graph.MaxPool(input, new long[] { 1, FilterHeight, FilterWidth, 1 },
                new long[] { 1, StrideHeight, StrideWidth, 1 }, Padding.ToString().ToUpperInvariant()), FusedActivationFunction);
        }
    }
}
