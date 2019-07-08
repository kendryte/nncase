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
        private void ConvertReshape(tflite.Operator op)
        {
            var input = GetTensor(op.Inputs(0));
            var options = op.BuiltinOptions<tflite.ReshapeOptions>().Value;

            var reshape = _graph.AddNode(new Reshape(ToDataType(input.Type), GetShape(input), options.GetNewShapeSpan()));

            _inputTensors.Add(reshape.Input, op.Inputs(0));
            _outputTensors.Add(op.Outputs(0), reshape.Output);
        }
    }
}
