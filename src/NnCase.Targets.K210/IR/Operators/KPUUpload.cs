using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;
using NnCase.IR;

namespace NnCase.Targets.K210.IR.FakeOperators
{
    public class KPUUpload : Node
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public KPUUpload(Shape inputShape)
        {
            Input = AddInput("input", DataType.UInt8, inputShape);
            Output = AddOutput("output", DataType.UInt8, inputShape, MemoryType.K210KPU);
        }
    }
}
