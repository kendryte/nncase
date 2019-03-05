using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.K210.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    public class K210RemovePaddingLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint Channels { get; set; }
    }

    [LayerConverter(typeof(K210RemovePadding), K210LayerType.K210RemovePadding)]
    public class K210RemovePaddingConverter
    {
        public K210RemovePaddingLayerArgument Convert(K210RemovePadding layer, ConvertContext context)
        {
            return new K210RemovePaddingLayerArgument
            {
                Channels = (uint)layer.Input.Dimensions[1]
            };
        }

        public void Infer(K210RemovePadding layer, K210RemovePaddingLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }
    }
}
