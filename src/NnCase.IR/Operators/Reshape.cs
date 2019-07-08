using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR.Operators
{
    public class Reshape : Node
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public Shape NewShape { get; }

        public Reshape(DataType type, Shape inputShape, Shape newShape)
        {
            NewShape = ShapeUtility.NormalizeReshape(inputShape, newShape);
            Input = AddInput("input", type, inputShape);
            Output = AddOutput("output", type, NewShape);
        }
    }
}
