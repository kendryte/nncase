using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.IR.Operators
{
    public class Conv2D : Node
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public DenseTensor<float> Weights { get; }

        public DenseTensor<float> Bias { get; }

        public int Groups { get; }

        public Padding PaddingH { get; }

        public Padding PaddingW { get; }

        public int StrideH { get; }

        public int StrideW { get; }

        public int DilationH { get; }

        public int DilationW { get; }

        public ValueRange<float> FusedActivation { get; }

        public Conv2D(Shape inputShape, DenseTensor<float> weights, DenseTensor<float> bias, int groups, Padding paddingH, Padding paddingW, int strideH, int strideW, int dilationH, int dilationW, ValueRange<float> fusedActivation)
        {
            Weights = weights;
            Bias = bias;
            Groups = groups;
            PaddingH = paddingH;
            PaddingW = paddingW;
            StrideH = strideH;
            StrideW = strideW;
            DilationH = dilationH;
            DilationW = dilationW;
            FusedActivation = fusedActivation;

            var outputShape = new Shape(
                inputShape[0],
                weights.Dimensions[0],
                ShapeUtility.GetWindowedOutputSize(inputShape[2] + paddingH.Sum, weights.Dimensions[2], strideH, dilationH, false),
                ShapeUtility.GetWindowedOutputSize(inputShape[3] + paddingW.Sum, weights.Dimensions[3], strideW, dilationW, false));

            Input = AddInput("input", DataType.Float32, inputShape);
            Output = AddOutput("output", DataType.Float32, outputShape);
        }
    }
}
