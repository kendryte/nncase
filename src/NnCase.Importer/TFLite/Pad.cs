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
    /// TFLite importer pad ops lowering.
    /// </summary>
    public partial class TFLiteImporter
    {
        private void ConvertPad(tflite.Operator op)
        {
            var input = GetTensor(op.Inputs(0));
            var paddings = LoadTensor<int>(GetTensor(op.Inputs(1)));
            var options = op.BuiltinOptions<tflite.PadOptions>().Value;

            var newPaddings = from i in Enumerable.Range(0, paddings.Dimensions[0])
                              select new Padding { Before = paddings[i, 0], After = paddings[i, 1] };

            var pad = _graph.AddNode(new Pad(ToDataType(input.Type), GetShape(input), newPaddings.ToList(), 0.0f));

            _inputTensors.Add(pad.Input, op.Inputs(0));
            _outputTensors.Add(op.Outputs(0), pad.Output);
        }
    }
}
