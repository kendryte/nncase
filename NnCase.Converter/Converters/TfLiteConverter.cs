using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Text;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.Converters
{
    public class TfLiteConverter
    {
        private readonly tflite.Model _model;
        private readonly tflite.SubGraph _graph;

        public TfLiteConverter(tflite.Model model, tflite.SubGraph graph)
        {
            _model = model;
            _graph = graph;
        }

        public void Convert()
        {
            var layers = _graph.GetOperators().Select(ConvertOperator).ToList();
        }

        private Layer ConvertOperator(tflite.Operator op)
        {
            var opCode = _model.OperatorCodes((int)op.OpcodeIndex).Value;
            switch (opCode.BuiltinCode)
            {
                case tflite.BuiltinOperator.SPACE_TO_BATCH_ND:
                    return ConvertSpaceToBatchNd(op);
                case tflite.BuiltinOperator.CONV_2D:
                    return ConvertConv2d(op);
                default:
                    throw new NotSupportedException();
            }
        }

        private Layer ConvertSpaceToBatchNd(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var input = _graph.Tensors(inputs[0]).Value;
            var blockShape = _graph.Tensors(inputs[1]).Value;
            var paddings = _graph.Tensors(inputs[2]).Value;

            return new SpaceToBatchNd(input.GetShapeArray().ToNCHW(), _model.GetTensor<int>(blockShape), _model.GetTensor<int>(paddings));
        }

        private Layer ConvertConv2d(tflite.Operator op)
        {
            var inputs = op.GetInputsArray();
            var input = _graph.Tensors(inputs[0]).Value;
            var options = op.BuiltinOptions<tflite.Conv2DOptions>().Value;
            var weights = _graph.Tensors(inputs[1]).Value;
            var bias = _graph.Tensors(inputs[2]).Value;

            return new Conv2d(input.GetShapeArray().ToNCHW(),  _model.GetTensor<float>(weights).ToOIHW(), _model.GetTensor<float>(bias),
                options.Padding.ToPadding(), options.StrideW, options.StrideH, options.FusedActivationFunction.ToActivationFunction());
        }
    }

    static class TfLiteExtensions
    {
        public static IEnumerable<tflite.Operator> GetOperators(this tflite.SubGraph subGraph)
        {
            for (int i = 0; i < subGraph.OperatorsLength; i++)
                yield return subGraph.Operators(i).Value;
        }

        public static Tensor<T> GetTensor<T>(this tflite.Model model, tflite.Tensor tensor)
            where T : unmanaged
        {
            var buffer = model.Buffers((int)tensor.Buffer).Value;
            return new DenseTensor<T>(MemoryMarshal.Cast<byte, T>(buffer.GetDataBytes()).ToArray(), tensor.GetShapeArray());
        }

        public static int[] ToNCHW(this int[] shape)
        {
            return new[] { shape[0], shape[3], shape[1], shape[2] };
        }

        public static Tensor<T> ToOIHW<T>(this Tensor<T> weights)
        {
            return weights.Transpose<T>(new[] { 0, 3, 1, 2 });
        }

        public static Padding ToPadding(this tflite.Padding padding)
        {
            switch (padding)
            {
                case tflite.Padding.SAME:
                    return Padding.Same;
                case tflite.Padding.VALID:
                    return Padding.Valid;
                default:
                    throw new ArgumentOutOfRangeException(nameof(padding));
            }
        }

        public static ActivationFunctionType ToActivationFunction(this tflite.ActivationFunctionType activation)
        {
            switch (activation)
            {
                case tflite.ActivationFunctionType.NONE:
                    return ActivationFunctionType.Linear;
                case tflite.ActivationFunctionType.RELU:
                    return ActivationFunctionType.Relu;
                case tflite.ActivationFunctionType.RELU6:
                    return ActivationFunctionType.Relu6;
                default:
                    throw new NotSupportedException(nameof(activation));
            }
        }
    }
}
