using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class SpaceToBatchNd : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public Tensor<int> BlockShape { get; }

        public Tensor<int> Paddings { get; }

        public SpaceToBatchNd(ReadOnlySpan<int> dimensions, Tensor<int> blockShape, Tensor<int> paddings)
        {
            BlockShape = blockShape;
            Paddings = paddings;

            Input = AddInput("input", dimensions);
            Output = AddOutput("output", new[] { dimensions[0], dimensions[1], dimensions[2] + 2, dimensions[3] + 2 });
        }
    }
}
