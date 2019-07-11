using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR.Operators
{
    public class Pad : Node
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public IReadOnlyList<Padding> Paddings { get; }

        public Scalar PadValue { get; }

        public Pad(DataType type, Shape inputShape, IReadOnlyList<Padding> paddings, Scalar padValue)
        {
            Paddings = paddings;
            PadValue = padValue;
            Input = AddInput("input", type, inputShape);
            Output = AddOutput("output", type, ShapeUtility.GetPaddedShape(inputShape, paddings));
        }
    }
}
