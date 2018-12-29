using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace NnCase.Converter.Model.Layers
{
    public class InputLayer : Layer
    {
        public OutputConnector Output { get; }

        public string Name { get; set; }

        public InputLayer(ReadOnlySpan<int> dimensions)
        {
            Output = AddOutput("output", dimensions);
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            context.Inputs[this] = context.TFOutputs[Output] = context.TFGraph.Placeholder(TFDataType.Float, Output.Dimensions.ToNHWC(), Name);
        }
    }
}
