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
        private void ConvertConv2D(tflite.Operator op)
        {
            var input = GetTensor(op.Inputs(0));
            var weights = GetTensor(op.Inputs(1));
            var bias = GetTensor(op.Inputs(2));
            var options = op.BuiltinOptions<tflite.Conv2DOptions>().Value;

            var weightsTensor = LoadTensor<float>(weights).Transpose(new[] { 0, 3, 1, 2 });
            var biasTensor = LoadTensor<float>(bias);

            var preTrans = NHWCToNCHW(DataType.Float32, GetShape(input));

            var inH = preTrans.Output.Shape[2];
            var inW = preTrans.Output.Shape[3];
            var fH = weightsTensor.Dimensions[2];
            var fW = weightsTensor.Dimensions[3];
            var strideH = options.StrideH;
            var strideW = options.StrideW;
            var dilationH = options.DilationHFactor;
            var dilationW = options.DilationWFactor;
            var padH = ShapeUtility.GetWindowedPadding(inH, fH, strideH, dilationH, options.Padding == tflite.Padding.SAME);
            var padW = ShapeUtility.GetWindowedPadding(inW, fW, strideW, dilationW, options.Padding == tflite.Padding.SAME);
            var conv2d = _graph.AddNode(new Conv2D(preTrans.Output.Shape, weightsTensor, biasTensor, 1, padH, padW, strideH, strideW, dilationH, dilationW, ToFloatClampRange(options.FusedActivationFunction)));
            conv2d.Input.Connect(preTrans.Output);

            var surTrans = NCHWToNHWC(DataType.Float32, conv2d.Output.Shape);
            surTrans.Input.Connect(conv2d.Output);

            _inputTensors.Add(preTrans.Input, op.Inputs(0));
            _outputTensors.Add(op.Outputs(0), surTrans.Output);
        }

        private void ConvertDepthwiseConv2D(tflite.Operator op)
        {
            var input = GetTensor(op.Inputs(0));
            var weights = GetTensor(op.Inputs(1));
            var bias = GetTensor(op.Inputs(2));
            var options = op.BuiltinOptions<tflite.DepthwiseConv2DOptions>().Value;

            var weightsTensor = LoadTensor<float>(weights).Transpose(new[] { 3, 0, 1, 2 });
            var biasTensor = LoadTensor<float>(bias);

            var preTrans = NHWCToNCHW(DataType.Float32, GetShape(input));

            var inH = preTrans.Output.Shape[2];
            var inW = preTrans.Output.Shape[3];
            var groups = weightsTensor.Dimensions[0];
            var fH = weightsTensor.Dimensions[2];
            var fW = weightsTensor.Dimensions[3];
            var strideH = options.StrideH;
            var strideW = options.StrideW;
            var dilationH = options.DilationHFactor;
            var dilationW = options.DilationWFactor;
            var padH = ShapeUtility.GetWindowedPadding(inH, fH, strideH, dilationH, options.Padding == tflite.Padding.SAME);
            var padW = ShapeUtility.GetWindowedPadding(inW, fW, strideW, dilationW, options.Padding == tflite.Padding.SAME);
            var conv2d = _graph.AddNode(new Conv2D(preTrans.Output.Shape, weightsTensor, biasTensor, groups, padH, padW, strideH, strideW, dilationH, dilationW, ToFloatClampRange(options.FusedActivationFunction)));
            conv2d.Input.Connect(preTrans.Output);

            var surTrans = NCHWToNHWC(DataType.Float32, conv2d.Output.Shape);
            surTrans.Input.Connect(conv2d.Output);

            _inputTensors.Add(preTrans.Input, op.Inputs(0));
            _outputTensors.Add(op.Outputs(0), surTrans.Output);
        }

        private Transpose NHWCToNCHW(DataType type, Shape shape)
        {
            return _graph.AddNode(new Transpose(type, shape, new[] { 0, 3, 1, 2 }));
        }

        private Transpose NCHWToNHWC(DataType type, Shape shape)
        {
            return _graph.AddNode(new Transpose(type, shape, new[] { 0, 2, 3, 1 }));
        }
    }
}
