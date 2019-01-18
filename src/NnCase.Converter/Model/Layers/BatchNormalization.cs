using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class BatchNormalization : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public Tensor<float> Scale { get; }

        public Tensor<float> Offset { get; }

        public Tensor<float> Mean { get; }

        public Tensor<float> Variance { get; }

        public float Epsilon { get; }

        public BatchNormalization(ReadOnlySpan<int> dimensions, Tensor<float> scale, Tensor<float> offset, Tensor<float> mean, Tensor<float> variance, float epsilon)
        {
            Scale = scale;
            Offset = offset;
            Mean = mean;
            Variance = variance;
            Epsilon = epsilon;

            Input = AddInput("input", dimensions);
            Output = AddOutput("output", dimensions);
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;
            var x = context.TFOutputs[Input.Connection.From];
            var scale = Scale.ToNHWC();
            var offset = Offset.ToNHWC();
            var mean = Mean.ToNHWC();
            var variance = Variance.ToNHWC();

            if (Input.Dimensions.Length == 4)
            {
                context.TFOutputs[Output] = graph.FusedBatchNorm(x, graph.Const(scale), graph.Const(offset), graph.Const(mean), graph.Const(variance), Epsilon, is_training: false).y;
            }
            else
            {
                throw new NotImplementedException();
            }
        }
    }
}
