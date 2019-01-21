using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class PRelu : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public Tensor<float> Slope { get; }

        public PRelu(ReadOnlySpan<int> dimensions, Tensor<float> slope)
        {
            Input = AddInput("input", dimensions);
            Output = AddOutput("output", dimensions);
            Slope = slope;
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var x = context.TFOutputs[Input.Connection.From];
            var alpha = graph.Const(Slope.ToNHWC());
            var zero = graph.Const(0.0f, TensorFlow.TFDataType.Float);

            var pos = graph.Maximum(zero, x);
            var neg = graph.Mul(alpha, graph.Minimum(zero, x));

            context.TFOutputs[Output] = graph.Add(pos, neg);
        }
    }
}
