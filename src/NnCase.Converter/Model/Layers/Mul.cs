using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class Mul : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public object Scale { get; }

        public Mul(ReadOnlySpan<int> dimensions, Tensor<float> scale)
        {
            Input = AddInput("input", dimensions);
            Output = AddOutput("output", dimensions);
            Scale = scale;
        }

        public Mul(ReadOnlySpan<int> dimensions, float scale)
        {
            Input = AddInput("input", dimensions);
            Output = AddOutput("output", dimensions);
            Scale = scale;
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var input = context.TFOutputs[Input.Connection.From];

            if (Scale is Tensor<float> tensorScale)
                context.TFOutputs[Output] = graph.Mul(input, graph.Const(tensorScale.ToNHWC()));
            else if(Scale is float scalarScale)
                context.TFOutputs[Output] = graph.Mul(input, graph.Const(scalarScale));
        }
    }
}
