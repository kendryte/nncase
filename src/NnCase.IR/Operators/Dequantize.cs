using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR.Operators
{
    public class Dequantize : Node
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public QuantizationParam QuantizationParam { get; }

        public Dequantize(Shape inputShape, QuantizationParam quantizationParam)
        {
            QuantizationParam = quantizationParam;

            Input = AddInput("input", DataType.UInt8, inputShape);
            Output = AddOutput("output", DataType.Float32, inputShape);
        }
    }
}
