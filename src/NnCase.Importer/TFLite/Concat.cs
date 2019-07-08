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
    /// TFLite importer convolution ops lowering.
    /// </summary>
    public partial class TFLiteImporter
    {
        private void ConvertConcatenation(tflite.Operator op)
        {
            var inputs = Vector(op.Inputs, op.InputsLength).Select(GetTensor).ToList();
            var options = op.BuiltinOptions<tflite.ConcatenationOptions>().Value;

            if (options.FusedActivationFunction != tflite.ActivationFunctionType.NONE)
                throw new NotSupportedException("Concat doesn't support activations");

            var concat = _graph.AddNode(new Concat(ToDataType(inputs[0].Type), inputs.Select(GetShape), options.Axis));

            for (int i = 0; i < op.InputsLength; i++)
                _inputTensors.Add(concat.Inputs[i], op.Inputs(i));
            _outputTensors.Add(op.Outputs(0), concat.Output);
        }
    }
}
