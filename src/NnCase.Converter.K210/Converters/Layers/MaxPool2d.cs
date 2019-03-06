using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    public class MaxPool2dLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint Width { get; set; }

        public uint Height { get; set; }

        public uint Channels { get; set; }

        public uint KernelWidth { get; set; }

        public uint KernelHeight { get; set; }

        public uint StrideWidth { get; set; }

        public uint StrideHeight { get; set; }

        public Padding Padding { get; set; }

        public ActivationFunctionType Activation { get; set; }
    }

    [LayerConverter(typeof(MaxPool2d), K210LayerType.MaxPool2d)]
    public class MaxPool2dConverter
    {
        public MaxPool2dLayerArgument Convert(MaxPool2d layer, ConvertContext context)
        {
            return new MaxPool2dLayerArgument
            {
                Width = (uint)layer.Input.Dimensions[3],
                Height = (uint)layer.Input.Dimensions[2],
                Channels = (uint)layer.Input.Dimensions[1],
                KernelWidth = (uint)layer.FilterWidth,
                KernelHeight = (uint)layer.FilterHeight,
                StrideWidth = (uint)layer.StrideWidth,
                StrideHeight = (uint)layer.StrideHeight,
                Padding = layer.Padding,
                Activation = layer.FusedActivationFunction
            };
        }

        public void Infer(MaxPool2d layer, MaxPool2dLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }
    }
}
