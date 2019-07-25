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
    /// TFLite importer softmax lowering.
    /// </summary>
    public partial class TFLiteImporter
    {
        private void ConvertSoftmax(tflite.Operator op)
        {
            var input = GetTensor(op.Inputs(0));
            var options = op.BuiltinOptions<tflite.SoftmaxOptions>().Value;

            var inputShape = GetShape(input);
            var reduceAxis = inputShape.Count == 1 ? new[] { 0 } : Enumerable.Range(1, inputShape.Count - 1).ToArray();
            var max = _graph.AddNode(new Reduce(ReduceOperator.Max, float.MinValue, inputShape, reduceAxis, false));
            var sub = _graph.AddNode(new Binary(BinaryOperator.Sub, inputShape, max.Output.Shape, ValueRanges.DefaultFloat));
            var beta = _graph.AddNode(new Constant(DataType.Float32, BitConverter.GetBytes(options.Beta), new Shape(1)));
            var mul = _graph.AddNode(new Binary(BinaryOperator.Mul, sub.Output.Shape, beta.Shape, ValueRanges.DefaultFloat));
            var exp = _graph.AddNode(new Unary(UnaryOperator.Exp, mul.Output.Shape));
            var sum = _graph.AddNode(new Reduce(ReduceOperator.Sum, 0, exp.Output.Shape, reduceAxis, false));
            var div = _graph.AddNode(new Binary(BinaryOperator.Div, exp.Output.Shape, sum.Output.Shape, ValueRanges.DefaultFloat));

            sub.InputB.Connect(max.Output);
            mul.InputA.Connect(sub.Output);
            mul.InputB.Connect(beta.Output);
            exp.Input.Connect(mul.Output);
            sum.Input.Connect(exp.Output);
            div.InputA.Connect(exp.Output);
            div.InputB.Connect(sum.Output);

            _inputTensors.Add(max.Input, op.Inputs(0));
            _inputTensors.Add(sub.InputA, op.Inputs(0));
            _outputTensors.Add(op.Outputs(0), div.Output);
        }
    }
}
