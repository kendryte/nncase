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
        private void ConvertMaxPool2D(tflite.Operator op)
        {
            ConvertPool2D(op, ReduceOperator.Max, float.MinValue);
        }

        private void ConvertAveragePool2D(tflite.Operator op)
        {
            ConvertPool2D(op, ReduceOperator.Mean, 0);
        }

        private void ConvertPool2D(tflite.Operator op, ReduceOperator reduceOperator, float initialValue)
        {
            var input = GetTensor(op.Inputs(0));
            var options = op.BuiltinOptions<tflite.Pool2DOptions>().Value;

            var preTrans = NHWCToNCHW(DataType.Float32, GetShape(input));

            var inH = preTrans.Output.Shape[2];
            var inW = preTrans.Output.Shape[3];
            var fH = options.FilterHeight;
            var fW = options.FilterWidth;
            var strideH = options.StrideH;
            var strideW = options.StrideW;
            var dilationH = 1;
            var dilationW = 1;
            var padH = ShapeUtility.GetWindowedPadding(inH, fH, strideH, dilationH, options.Padding == tflite.Padding.SAME);
            var padW = ShapeUtility.GetWindowedPadding(inW, fW, strideW, dilationW, options.Padding == tflite.Padding.SAME);
            var conv2d = _graph.AddNode(new ReduceWindow2D(reduceOperator, initialValue, preTrans.Output.Shape, fH, fW, padH, padW, strideH, strideW, dilationH, dilationW, ToFloatClampRange(options.FusedActivationFunction)));
            conv2d.Input.Connect(preTrans.Output);

            var surTrans = NCHWToNHWC(DataType.Float32, conv2d.Output.Shape);
            surTrans.Input.Connect(conv2d.Output);

            _inputTensors.Add(preTrans.Input, op.Inputs(0));
            _outputTensors.Add(op.Outputs(0), surTrans.Output);
        }
    }
}
