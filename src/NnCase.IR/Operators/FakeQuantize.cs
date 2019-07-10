using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR.Operators
{
    public class FakeQuantize : Node
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public FakeQuantize(Shape inputShape)
        {
            Input = AddInput("input", DataType.Float32, inputShape);
            Output = AddOutput("output", DataType.Float32, inputShape);
        }
    }
}
