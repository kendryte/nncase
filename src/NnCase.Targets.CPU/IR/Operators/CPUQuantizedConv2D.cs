using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;
using NnCase.IR;

namespace NnCase.Targets.CPU.IR.Operators
{
    public class CPUQuantizedConv2D : Node
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public DenseTensor<byte> Weights { get; }

        public DenseTensor<int> Bias { get; }

        public Padding PaddingH { get; }

        public Padding PaddingW { get; }

        public int StrideH { get; }

        public int StrideW { get; }

        public int DilationH { get; }

        public int DilationW { get; }

        public int InputOffset { get; }

        public int FilterOffset { get; }

        public int OutputMul { get; }

        public int OutputShift { get; }

        public int OutputOffset { get; }

        public CPUQuantizedConv2D(Shape inputShape, DenseTensor<byte> weights, DenseTensor<int> bias, Padding paddingH, Padding paddingW, int strideH, int strideW, int dilationH, int dilationW, int inputOffset, int filterOffset, int outputMul, int outputShift, int outputOffset)
        {
            Weights = weights;
            Bias = bias;
            PaddingH = paddingH;
            PaddingW = paddingW;
            StrideH = strideH;
            StrideW = strideW;
            DilationH = dilationH;
            DilationW = dilationW;
            InputOffset = inputOffset;
            FilterOffset = filterOffset;
            OutputMul = outputMul;
            OutputShift = outputShift;
            OutputOffset = outputOffset;

            var outputShape = new Shape(
                inputShape[0],
                ShapeUtility.GetWindowedOutputSize(inputShape[1] + paddingH.Sum, weights.Dimensions[1], strideH, dilationH, false),
                ShapeUtility.GetWindowedOutputSize(inputShape[2] + paddingW.Sum, weights.Dimensions[2], strideW, dilationW, false),
                weights.Dimensions[0]);

            Input = AddInput("input", DataType.UInt8, inputShape);
            Output = AddOutput("output", DataType.UInt8, outputShape);
        }
    }
}
