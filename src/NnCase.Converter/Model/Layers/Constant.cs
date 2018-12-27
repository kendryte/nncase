using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;
using TensorFlow;

namespace NnCase.Converter.Model.Layers
{
    public class Constant : Layer
    {
        public OutputConnector Output { get; }

        public Tensor<float> Value { get; }

        public Constant(ReadOnlySpan<int> dimensions, Tensor<float> value)
        {
            Value = value;
            Output = AddOutput("output", dimensions);
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            context.TFOutputs[Output] = context.TFGraph.Const(Value.ToNHWC(), TFDataType.Float);
        }
    }
}
