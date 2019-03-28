using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.K210.Converters.Stages.Quantize;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    public class QuantizedMaxPool2dLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint InputWidth { get; set; }

        public uint InputHeight { get; set; }

        public uint InputChannels { get; set; }

        public uint OutputWidth { get; set; }

        public uint OutputHeight { get; set; }

        public uint OutputChannels { get; set; }

        public uint KernelWidth { get; set; }

        public uint KernelHeight { get; set; }

        public uint StrideWidth { get; set; }

        public uint StrideHeight { get; set; }

        public uint PaddingWidth { get; set; }

        public uint PaddingHeight { get; set; }
    }

    [LayerConverter(typeof(QuantizedMaxPool2d), K210LayerType.QuantizedMaxPool2d)]
    public class QuantizedMaxPool2dConverter
    {
        public QuantizedMaxPool2dLayerArgument Convert(QuantizedMaxPool2d layer, ConvertContext context)
        {
            var inputRange = context.Quantization.Distributions[layer.Input.Connection.From].Global;
            var outputRange = context.Quantization.Distributions[layer.Output].Global;

            (var sa, var ba) = inputRange.GetScaleBias(8);
            (var so, var bo) = outputRange.GetScaleBias(8);

            (var mulO, var shiftO) = Quantizer.ExtractValueAndShift(so / sa, 32, 32);

            return new QuantizedMaxPool2dLayerArgument
            {
                InputWidth = (uint)layer.Input.Dimensions[3],
                InputHeight = (uint)layer.Input.Dimensions[2],
                InputChannels = (uint)layer.Input.Dimensions[1],
                OutputWidth = (uint)layer.Output.Dimensions[3],
                OutputHeight = (uint)layer.Output.Dimensions[2],
                OutputChannels = (uint)layer.Output.Dimensions[1],
                KernelWidth = (uint)layer.FilterWidth,
                KernelHeight = (uint)layer.FilterHeight,
                StrideWidth = (uint)layer.StrideWidth,
                StrideHeight = (uint)layer.StrideHeight,
                PaddingWidth = (uint)Layer.GetPadding(layer.Input.Dimensions[3], layer.Output.Dimensions[3], layer.StrideWidth, 1, layer.FilterWidth),
                PaddingHeight = (uint)Layer.GetPadding(layer.Input.Dimensions[2], layer.Output.Dimensions[2], layer.StrideHeight, 1, layer.FilterHeight)
            };
        }

        public void Infer(QuantizedMaxPool2d layer, QuantizedMaxPool2dLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }
    }
}
