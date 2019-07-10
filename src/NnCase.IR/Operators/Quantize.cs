using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR.Operators
{
    public class Quantize : Node
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public QuantizationParam QuantizationParam { get; }

        public Quantize(Shape inputShape, QuantizationParam quantizationParam)
        {
            QuantizationParam = quantizationParam;

            Input = AddInput("input", DataType.Float32, inputShape);
            Output = AddOutput("output", DataType.UInt8, inputShape);
        }
    }
}
