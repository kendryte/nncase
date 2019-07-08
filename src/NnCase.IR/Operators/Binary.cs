using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.IR.Operators
{
    public class Binary : Node
    {
        public InputConnector InputA { get; }

        public InputConnector InputB { get; }

        public OutputConnector Output { get; }

        public BinaryOperator BinaryOperator { get; }

        public ValueRange<float> FusedActivation { get; }

        public Binary(BinaryOperator binaryOperator, Shape inputAShape, Shape inputBShape, ValueRange<float> fusedActivation)
        {
            BinaryOperator = binaryOperator;
            FusedActivation = fusedActivation;

            InputA = AddInput("inputA", DataType.Float32, inputAShape);
            InputB = AddInput("inputB", DataType.Float32, inputBShape);
            Output = AddOutput("output", DataType.Float32, ShapeUtility.GetBinaryOutputShape(inputAShape, inputBShape));
        }
    }
}
