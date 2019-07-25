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
        private void ConvertAbs(tflite.Operator op)
            => ConvertUnary(op, UnaryOperator.Abs);

        private void ConvertCeil(tflite.Operator op)
            => ConvertUnary(op, UnaryOperator.Ceil);

        private void ConvertCos(tflite.Operator op)
            => ConvertUnary(op, UnaryOperator.Cos);

        private void ConvertExp(tflite.Operator op)
            => ConvertUnary(op, UnaryOperator.Exp);

        private void ConvertFloor(tflite.Operator op)
            => ConvertUnary(op, UnaryOperator.Floor);

        private void ConvertLog(tflite.Operator op)
            => ConvertUnary(op, UnaryOperator.Log);

        private void ConvertNeg(tflite.Operator op)
            => ConvertUnary(op, UnaryOperator.Neg);

        private void ConvertRsqrt(tflite.Operator op)
            => ConvertUnary(op, UnaryOperator.Rsqrt);

        private void ConvertSin(tflite.Operator op)
            => ConvertUnary(op, UnaryOperator.Sin);

        private void ConvertSqrt(tflite.Operator op)
            => ConvertUnary(op, UnaryOperator.Sqrt);

        private void ConvertUnary(tflite.Operator op, UnaryOperator unaryOperator)
        {
            var input = GetTensor(op.Inputs(0));

            var binary = _graph.AddNode(new Unary(unaryOperator, GetShape(input)));

            _inputTensors.Add(binary.Input, op.Inputs(0));
            _outputTensors.Add(op.Outputs(0), binary.Output);
        }
    }
}
