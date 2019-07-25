using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Importer
{
    /// <summary>
    /// TFLite importer LeakyRelu ops lowering.
    /// </summary>
    public partial class TFLiteImporter
    {
        private void ConvertLeakyRelu(tflite.Operator op)
        {
            var input = GetTensor(op.Inputs(0));
            var options = op.BuiltinOptions<tflite.LeakyReluOptions>().Value;

            var slope = _graph.AddNode(new Constant(DataType.Float32, BitConverter.GetBytes(options.Alpha), new Shape(1)));
            var mul = _graph.AddNode(new Binary(BinaryOperator.Mul, GetShape(input), slope.Shape, ValueRanges.DefaultFloat));
            var max = _graph.AddNode(new Binary(BinaryOperator.Max, GetShape(input), mul.Output.Shape, ValueRanges.DefaultFloat));
            mul.InputB.Connect(slope.Output);
            max.InputB.Connect(mul.Output);

            _inputTensors.Add(mul.InputA, op.Inputs(0));
            _inputTensors.Add(max.InputA, op.Inputs(0));
            _outputTensors.Add(op.Outputs(0), max.Output);
        }
    }
}
