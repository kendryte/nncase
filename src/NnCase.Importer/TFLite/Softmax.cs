using System;
using System.Collections.Generic;
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

            var softmax = _graph.AddNode(new Softmax(GetShape(input), options.Beta));

            _inputTensors.Add(softmax.Input, op.Inputs(0));
            _outputTensors.Add(op.Outputs(0), softmax.Output);
        }
    }
}
