using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.IR.Operators
{
    public class MatMul : Node
    {
        public InputConnector InputA { get; }

        public InputConnector InputB { get; }

        public OutputConnector Output { get; }

        public DenseTensor<float> Bias { get; }

        public ValueRange<float> FusedActivation { get; }

        public MatMul(Shape inputAShape, Shape inputBShape, DenseTensor<float> bias, ValueRange<float> fusedActivation)
        {
            Bias = bias;
            FusedActivation = fusedActivation;

            InputA = AddInput("inputA", DataType.Float32, inputAShape);
            InputB = AddInput("inputB", DataType.Float32, inputBShape);
            Output = AddOutput("output", DataType.Float32, new[] { inputAShape[0], inputBShape[1] });
        }
    }
}
