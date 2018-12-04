using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class AveragePool2d : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public Padding Padding { get; }

        public int StrideWidth { get; }

        public int StrideHeight { get; }

        public ActivationFunctionType FusedActivationFunction { get; }

        public AveragePool2d(ReadOnlySpan<int> dimensions, Padding padding, int filterWidth, int filterHeight, int strideWidth, int strideHeight, ActivationFunctionType fusedActivationFunction)
        {
            Padding = padding;
            StrideWidth = strideWidth;
            StrideHeight = strideHeight;
            FusedActivationFunction = fusedActivationFunction;

            Input = AddInput("input", dimensions);
            Output = AddOutput("output", new[] {
                dimensions[0],
                (dimensions[1] - (padding == Padding.Valid ? filterHeight - 1 : 0)) / strideHeight,
                (dimensions[2] - (padding == Padding.Valid ? filterWidth - 1 : 0)) / strideWidth,
                dimensions[3]
            });
        }
    }
}
