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
        private void ConvertMean(tflite.Operator op)
        {
            ConvertReduce(op, ReduceOperator.Mean, 0);
        }

        private void ConvertReduceMin(tflite.Operator op)
        {
            ConvertReduce(op, ReduceOperator.Min, float.MaxValue);
        }

        private void ConvertReduceMax(tflite.Operator op)
        {
            ConvertReduce(op, ReduceOperator.Max, float.MaxValue);
        }

        private void ConvertReduceSum(tflite.Operator op)
        {
            ConvertReduce(op, ReduceOperator.Sum, 0);
        }

        private void ConvertReduce(tflite.Operator op, ReduceOperator reduceOperator, float initialValue)
        {
            var input = GetTensor(op.Inputs(0));
            var axis = new Shape(LoadTensor<int>(GetTensor(op.Inputs(1))));
            var options = op.BuiltinOptions<tflite.ReducerOptions>().Value;

            var binary = _graph.AddNode(new Reduce(reduceOperator, initialValue, GetShape(input), axis, options.KeepDims));

            _inputTensors.Add(binary.Input, op.Inputs(0));
            _outputTensors.Add(op.Outputs(0), binary.Output);
        }
    }
}
