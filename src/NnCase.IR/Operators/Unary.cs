using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.IR.Operators
{
    public class Unary : Node
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public UnaryOperator UnaryOperator { get; }

        public Unary(UnaryOperator unaryOperator, Shape inputShape)
        {
            UnaryOperator = unaryOperator;

            Input = AddInput("input", DataType.Float32, inputShape);
            Output = AddOutput("output", DataType.Float32, inputShape);
        }
    }
}
