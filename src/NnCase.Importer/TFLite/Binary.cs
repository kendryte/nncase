using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Importer
{
    /// <summary>
    /// TFLite importer convolution ops lowering.
    /// </summary>
    public partial class TFLiteImporter
    {
        private void ConvertAdd(tflite.Operator op)
        {
            ConvertBinary(op, BinaryOperator.Add, op.BuiltinOptions<tflite.AddOptions>().Value.FusedActivationFunction);
        }

        private void ConvertSub(tflite.Operator op)
        {
            ConvertBinary(op, BinaryOperator.Sub, op.BuiltinOptions<tflite.SubOptions>().Value.FusedActivationFunction);
        }

        private void ConvertMul(tflite.Operator op)
        {
            ConvertBinary(op, BinaryOperator.Mul, op.BuiltinOptions<tflite.MulOptions>().Value.FusedActivationFunction);
        }

        private void ConvertDiv(tflite.Operator op)
        {
            ConvertBinary(op, BinaryOperator.Div, op.BuiltinOptions<tflite.DivOptions>().Value.FusedActivationFunction);
        }

        private void ConvertBinary(tflite.Operator op, BinaryOperator binaryOperator, tflite.ActivationFunctionType activationFunctionType)
        {
            var inputA = GetTensor(op.Inputs(0));
            var inputB = GetTensor(op.Inputs(1));

            if (inputA.Type != inputB.Type)
                throw new ArgumentException($"Inputs of {binaryOperator} must have same types");

            var binary = _graph.AddNode(new Binary(binaryOperator, GetShape(inputA), GetShape(inputB), ToFloatClampRange(activationFunctionType)));

            _inputTensors.Add(binary.InputA, op.Inputs(0));
            _inputTensors.Add(binary.InputB, op.Inputs(1));
            _outputTensors.Add(op.Outputs(0), binary.Output);
        }
    }
}
