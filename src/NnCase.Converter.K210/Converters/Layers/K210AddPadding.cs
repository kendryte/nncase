using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.K210.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    public class K210AddPaddingLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint KPUMemoryOutputAddress { get; set; }

        public uint Channels { get; set; }
    }

    [LayerConverter(typeof(K210AddPadding), K210LayerType.K210AddPadding)]
    public class K210AddPaddingConverter
    {
        public K210AddPaddingLayerArgument Convert(K210AddPadding layer, ConvertContext context)
        {
            return new K210AddPaddingLayerArgument
            {
                Channels = (uint)layer.Input.Dimensions[1]
            };
        }

        public void Infer(K210AddPadding layer, K210AddPaddingLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.KPUMemoryMap[layer.Output];

            argument.Flags = K210LayerFlags.None;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.KPUMemoryOutputAddress = outputAlloc.GetAddress();
        }
    }
}
