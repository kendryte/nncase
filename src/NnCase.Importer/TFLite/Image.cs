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
        private void ConvertResizeNearestNeighbor(tflite.Operator op)
        {
            var input = GetTensor(op.Inputs(0));
            var outSize = LoadTensor<int>(GetTensor(op.Inputs(1)));
            var options = op.BuiltinOptions<tflite.ResizeNearestNeighborOptions>().Value;
            var dataType = ToDataType(input.Type);

            var preTrans = NHWCToNCHW(dataType, GetShape(input));

            var outH = outSize[0];
            var outW = outSize[1];
            var resize = _graph.AddNode(new ResizeNearestNeighbor(dataType, preTrans.Output.Shape, outH, outW, options.AlignCorners));
            resize.Input.Connect(preTrans.Output);

            var surTrans = NCHWToNHWC(dataType, resize.Output.Shape);
            surTrans.Input.Connect(resize.Output);

            _inputTensors.Add(preTrans.Input, op.Inputs(0));
            _outputTensors.Add(op.Outputs(0), surTrans.Output);
        }

        private void ConvertResizeBilinear(tflite.Operator op)
        {
            var input = GetTensor(op.Inputs(0));
            var outSize = LoadTensor<int>(GetTensor(op.Inputs(1)));
            var options = op.BuiltinOptions<tflite.ResizeBilinearOptions>().Value;
            var dataType = ToDataType(input.Type);

            var preTrans = NHWCToNCHW(dataType, GetShape(input));

            var outH = outSize[0];
            var outW = outSize[1];
            var resize = _graph.AddNode(new ResizeBilinear(dataType, preTrans.Output.Shape, outH, outW, options.AlignCorners));
            resize.Input.Connect(preTrans.Output);

            var surTrans = NCHWToNHWC(dataType, resize.Output.Shape);
            surTrans.Input.Connect(resize.Output);

            _inputTensors.Add(preTrans.Input, op.Inputs(0));
            _outputTensors.Add(op.Outputs(0), surTrans.Output);
        }
    }
}
