using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR.Operators
{
    public class Constant : Node
    {
        public OutputConnector Output { get; }

        public ReadOnlyMemory<byte> Data { get; }

        public Shape Shape { get; }

        public Constant(DataType type, ReadOnlyMemory<byte> data, Shape shape)
        {
            Data = data;
            Shape = shape;
            Output = AddOutput("output", type, shape);
        }
    }
}
