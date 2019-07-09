using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.IR.Operators
{
    public class ResizeNearestNeighbor : Node
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public int OutputWidth { get; }

        public int OutputHeight { get; }

        public bool AlignCorners { get; }

        public ResizeNearestNeighbor(DataType type, Shape inputShape, int outputHeight, int outputWidth, bool alignCorners)
        {
            OutputHeight = outputHeight;
            OutputWidth = outputWidth;
            AlignCorners = alignCorners;

            var outputShape = new Shape(
                inputShape[0],
                inputShape[1],
                outputHeight,
                outputWidth);

            Input = AddInput("input", type, inputShape);
            Output = AddOutput("output", type, outputShape);
        }
    }
}
