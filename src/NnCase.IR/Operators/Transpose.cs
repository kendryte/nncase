using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR.Operators
{
    public class Transpose : Node
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public Shape Perm { get; }

        public Transpose(DataType type, Shape inputShape, Shape perm)
        {
            Perm = perm;
            Input = AddInput("input", type, inputShape);
            Output = AddOutput("output", type, ShapeUtility.GetTransposedShape(inputShape, perm));
        }
    }
}
