using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;
using NnCase.IR;

namespace NnCase.Targets.K210.IR.FakeOperators
{
    public class KPUConv2D : Node
    {
        public InputConnector Input { get; }

        public OutputConnector KPUOutput { get; }

        public OutputConnector MainMemoryOutput { get; }

        public bool IsDepthwise { get; }

        public KPUFilterType FilterType { get; }

        public KPUPoolType PoolType { get; }

        public DenseTensor<byte> Weights { get; }

        public byte PadValue { get; }

        public int ArgX { get; }

        public int ShiftX { get; }

        public int ArgW { get; }

        public int ShiftW { get; }

        public long ArgAdd { get; }

        public KPUBatchNormSegment[] BatchNorm { get; }

        public KPUActivationSegment[] Activation { get; }

        public KPUConv2D(Shape inputShape, bool isDepthwise, KPUFilterType filterType, KPUPoolType poolType, DenseTensor<byte> weights, byte padValue, int argX, int shiftX, int argW, int shiftW, long argAdd, KPUBatchNormSegment[] batchNorm, KPUActivationSegment[] activation, bool hasMainMemoryOutput)
        {
            IsDepthwise = isDepthwise;
            FilterType = filterType;
            PoolType = poolType;
            Weights = weights;
            PadValue = padValue;
            ArgX = argX;
            ShiftX = shiftX;
            ArgW = argW;
            ShiftW = shiftW;
            ArgAdd = argAdd;
            BatchNorm = batchNorm;
            Activation = activation;

            var outputShape = new Shape(
                inputShape[0],
                weights.Dimensions[0],
                KPUShapeUtility.GetKPUOutputSize(inputShape[2], poolType),
                KPUShapeUtility.GetKPUOutputSize(inputShape[3], poolType));

            Input = AddInput("input", DataType.UInt8, inputShape);
            KPUOutput = AddOutput("kpuOutput", DataType.UInt8, outputShape, MemoryType.K210KPU);
            if (hasMainMemoryOutput)
                MainMemoryOutput = AddOutput("mainMemoryOutput", DataType.UInt8, outputShape);
        }
    }
}
