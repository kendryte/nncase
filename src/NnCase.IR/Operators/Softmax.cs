using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.IR.Operators
{
    public class Softmax : Node
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public float Beta { get; }

        public Softmax(Shape inputShape, float beta)
        {
            Beta = beta;

            Input = AddInput("input", DataType.Float32, inputShape);
            Output = AddOutput("output", DataType.Float32, inputShape);
        }
    }
}
