using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.K210.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    public class K210UploadLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint KPUMemoryOutputAddress { get; set; }

        public uint Width { get; set; }

        public uint Height { get; set; }

        public uint Channels { get; set; }
    }

    [LayerConverter(typeof(K210Upload), K210LayerType.K210Upload)]
    public class K210UploadConverter
    {
        public K210UploadLayerArgument Convert(K210Upload layer, ConvertContext context)
        {
            return new K210UploadLayerArgument
            {
                Width = (uint)layer.Input.Dimensions[3],
                Height = (uint)layer.Input.Dimensions[2],
                Channels = (uint)layer.Input.Dimensions[1]
            };
        }

        public void Infer(K210Upload layer, K210UploadLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.KPUMemoryMap[layer.Output];

            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.KPUMemoryOutputAddress = outputAlloc.GetAddress();
        }
    }
}
