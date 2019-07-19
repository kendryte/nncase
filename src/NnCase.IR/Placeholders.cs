using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR
{
    public class InputNode : Node
    {
        public OutputConnector Output { get; }

        public InputNode(DataType type, Shape shape, MemoryType memoryType)
        {
            Output = AddOutput("output", type, shape, memoryType);
        }
    }

    public class OutputNode : Node
    {
        public InputConnector Input { get; }

        public OutputNode(DataType type, Shape shape)
        {
            Input = AddInput("input", type, shape);
        }
    }

    public class IgnoreNode : Node
    {
        public InputConnector Input { get; }

        public IgnoreNode(DataType type, Shape shape)
        {
            Input = AddInput("input", type, shape);
        }
    }
}
