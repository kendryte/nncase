using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Importer
{
    /// <summary>
    /// TFLite importer strided slice ops lowering.
    /// </summary>
    public partial class TFLiteImporter
    {
        private void ConvertStridedSlice(tflite.Operator op)
        {
            var input = GetTensor(op.Inputs(0));
            var begin = LoadTensor<int>(GetTensor(op.Inputs(1)));
            var end = LoadTensor<int>(GetTensor(op.Inputs(2)));
            var strides = LoadTensor<int>(GetTensor(op.Inputs(3)));
            var options = op.BuiltinOptions<tflite.StridedSliceOptions>().Value;

            var slice = _graph.AddNode(new StridedSlice(ToDataType(input.Type), GetShape(input), begin.Buffer, end.Buffer, strides.Buffer, options.BeginMask, options.EndMask, options.EllipsisMask, options.NewAxisMask, options.ShrinkAxisMask));

            _inputTensors.Add(slice.Input, op.Inputs(0));
            _outputTensors.Add(op.Outputs(0), slice.Output);
        }
    }
}
