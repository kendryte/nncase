using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    public class GlobalAveragePool2dLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint KernelSize { get; set; }

        public uint Channels { get; set; }
    }

    [LayerConverter(typeof(GlobalAveragePool), K210LayerType.GlobalAveragePool2d)]
    public class GlobalAveragePool2dConverter
    {
        public GlobalAveragePool2dLayerArgument Convert(GlobalAveragePool layer, ConvertContext context)
        {
            return new GlobalAveragePool2dLayerArgument
            {
                KernelSize = (uint)(layer.Input.Dimensions[2] * layer.Input.Dimensions[3]),
                Channels = (uint)(layer.Input.Dimensions[1])
            };
        }

        public void Infer(GlobalAveragePool layer, GlobalAveragePool2dLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }
    }
}
