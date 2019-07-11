using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;
using NnCase.IR;

namespace NnCase.Targets.K210.IR.FakeOperators
{
    public class KPUFakeConv2D : Node
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public bool IsDepthwise { get; }

        public KPUFilterType FilterType { get; }

        public KPUPoolType PoolType { get; }

        public DenseTensor<float> Weights { get; }

        public DenseTensor<float> Bias { get; }

        public ValueRange<float> FusedActivation { get; }

        public KPUFakeConv2D(Shape inputShape, bool isDepthwise, KPUFilterType filterType, KPUPoolType poolType, DenseTensor<float> weights, DenseTensor<float> bias, ValueRange<float> fusedActivation)
        {
            IsDepthwise = isDepthwise;
            FilterType = filterType;
            PoolType = poolType;
            Weights = weights;
            Bias = bias;
            FusedActivation = fusedActivation;

            var outputShape = new Shape(
                inputShape[0],
                weights.Dimensions[0],
                KPUShapeUtility.GetKPUOutputSize(inputShape[2], poolType),
                KPUShapeUtility.GetKPUOutputSize(inputShape[3], poolType));

            Input = AddInput("input", DataType.Float32, inputShape);
            Output = AddOutput("output", DataType.Float32, outputShape);
        }
    }
}
